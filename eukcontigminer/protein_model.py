"""Six-frame protein tokenization and an RC-invariant protein encoder."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


PAD_ID = 0
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: index + 1 for index, aa in enumerate(AA_ORDER)}
STOP_ID = 21
UNKNOWN_ID = 22
VOCABULARY_SIZE = 23

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _build_nt_lookup() -> np.ndarray:
    lookup = np.full(256, 4, dtype=np.uint8)
    for value, characters in enumerate((b"Aa", b"Cc", b"Gg", b"TtUu")):
        lookup[np.frombuffer(characters, dtype=np.uint8)] = value
    return lookup


def _build_codon_lookup() -> np.ndarray:
    lookup = np.full(125, UNKNOWN_ID, dtype=np.uint8)
    nt = {"A": 0, "C": 1, "G": 2, "T": 3}
    for codon, amino_acid in CODON_TABLE.items():
        code = nt[codon[0]] * 25 + nt[codon[1]] * 5 + nt[codon[2]]
        lookup[code] = STOP_ID if amino_acid == "*" else AA_TO_ID[amino_acid]
    return lookup


_NT_LOOKUP = _build_nt_lookup()
_CODON_LOOKUP = _build_codon_lookup()
_NT_COMPLEMENT = np.asarray([3, 2, 1, 0, 4], dtype=np.uint8)


def six_frame_token_count(length_bp: int) -> int:
    """Return the exact number of translated tokens retained for a sequence."""

    if type(length_bp) is not int or length_bp < 1:
        raise ValueError("length_bp must be a positive integer")
    one_strand = sum(max(0, (length_bp - offset) // 3) for offset in range(3))
    return 2 * one_strand


def translate_six_frames(sequence: str | bytes) -> tuple[np.ndarray, ...]:
    """Translate complete +0,+1,+2,-0,-1,-2 frames without ORF filtering.

    Stop codons and ambiguous codons are explicit tokens.  Therefore short
    peptides and noncoding sequence are retained rather than silently dropped.
    """

    if isinstance(sequence, str):
        raw = sequence.encode("ascii")
    elif isinstance(sequence, bytes):
        raw = sequence
    else:
        raise TypeError("sequence must be str or ASCII bytes")
    if not raw:
        raise ValueError("sequence must be non-empty")
    nucleotide = _NT_LOOKUP[np.frombuffer(raw, dtype=np.uint8)]
    reverse = _NT_COMPLEMENT[nucleotide][::-1]
    frames: list[np.ndarray] = []
    for strand in (nucleotide, reverse):
        for offset in range(3):
            codons = (len(strand) - offset) // 3
            if codons <= 0:
                frames.append(np.empty(0, dtype=np.uint8))
                continue
            stop = offset + 3 * codons
            first = strand[offset:stop:3].astype(np.int16, copy=False)
            second = strand[offset + 1 : stop : 3].astype(np.int16, copy=False)
            third = strand[offset + 2 : stop : 3].astype(np.int16, copy=False)
            code = first * 25 + second * 5 + third
            frames.append(_CODON_LOOKUP[code])
    result = tuple(frames)
    if len(result) != 6 or sum(map(len, result)) != six_frame_token_count(len(raw)):
        raise AssertionError("six-frame translation length invariant failed")
    return result


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, width: int, *, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=width,
        )
        self.norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, 4 * width)
        self.contract = nn.Linear(4 * width, width)

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = self.depthwise(values).transpose(1, 2)
        values = self.contract(F.gelu(self.expand(self.norm(values))))
        return residual + values.transpose(1, 2)


class SixFrameProteinEncoder(nn.Module):
    """Shared frame encoder with exact six-frame permutation invariance."""

    def __init__(
        self,
        *,
        embedding_dimension: int = 32,
        stem_width: int = 64,
        output_width: int = 96,
        frame_dimension: int = 128,
        dropout: float = 0.15,
        maximum_residual_logit: float = 4.0,
        stem_stride: int = 8,
        downsample_stride: int = 4,
    ) -> None:
        super().__init__()
        self.maximum_residual_logit = float(maximum_residual_logit)
        if not math.isfinite(self.maximum_residual_logit) or self.maximum_residual_logit <= 0:
            raise ValueError("maximum_residual_logit must be finite and positive")
        if stem_stride < 1 or downsample_stride < 1:
            raise ValueError("protein encoder strides must be positive")
        self.stem_stride = int(stem_stride)
        self.downsample_stride = int(downsample_stride)
        self.embedding = nn.Embedding(
            VOCABULARY_SIZE, embedding_dimension, padding_idx=PAD_ID
        )
        self.stem = nn.Conv1d(
            embedding_dimension,
            stem_width,
            kernel_size=15,
            stride=self.stem_stride,
            padding=7,
        )
        # Per-contig GroupNorm would make a score depend on how much padding a
        # neighbouring long contig introduces.  Position-wise normalization in
        # the ConvNeXt blocks avoids that variable-batch artefact.
        self.stem_norm = nn.Identity()
        self.stem_blocks = nn.ModuleList([
            ConvNeXt1DBlock(stem_width, kernel_size=5, dilation=1),
            ConvNeXt1DBlock(stem_width, kernel_size=5, dilation=2),
            ConvNeXt1DBlock(stem_width, kernel_size=5, dilation=4),
        ])
        self.downsample = nn.Conv1d(
            stem_width,
            output_width,
            kernel_size=7,
            stride=self.downsample_stride,
            padding=3,
        )
        self.output_norm = nn.Identity()
        self.output_blocks = nn.ModuleList([
            ConvNeXt1DBlock(output_width, kernel_size=7, dilation=1),
            ConvNeXt1DBlock(output_width, kernel_size=7, dilation=2),
            ConvNeXt1DBlock(output_width, kernel_size=7, dilation=4),
        ])
        self.attention = nn.Conv1d(output_width, 1, kernel_size=1)
        self.frame_projection = nn.Sequential(
            nn.Linear(3 * output_width, frame_dimension),
            nn.LayerNorm(frame_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        contig_dimension = 2 * frame_dimension + 6
        self.contig_projection = nn.Sequential(
            nn.Linear(contig_dimension, 2 * frame_dimension),
            nn.LayerNorm(2 * frame_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * frame_dimension, frame_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.protein_head = nn.Linear(frame_dimension, 1)
        self.residual_head = nn.Linear(frame_dimension, 1)

    @staticmethod
    def _downsample_lengths(lengths: Tensor, stride: int) -> Tensor:
        return torch.div(lengths + stride - 1, stride, rounding_mode="floor")

    def forward(self, tokens: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or tokens.shape[1] != 6:
            raise ValueError("tokens must have shape [batch,6,time]")
        if lengths.shape != tokens.shape[:2] or lengths.dtype != torch.long:
            raise ValueError("lengths must be torch.long with shape [batch,6]")
        if tokens.dtype != torch.long:
            raise ValueError("tokens must be torch.long")
        if torch.any(lengths < 1) or torch.any(lengths > tokens.shape[2]):
            raise ValueError("frame lengths lie outside the padded token tensor")
        if torch.any(tokens < 0) or torch.any(tokens >= VOCABULARY_SIZE):
            raise ValueError("protein token lies outside the vocabulary")

        batch, frames, time = tokens.shape
        flat_tokens = tokens.reshape(batch * frames, time)
        flat_lengths = lengths.reshape(-1)
        values = self.embedding(flat_tokens).transpose(1, 2)
        values = F.gelu(self.stem_norm(self.stem(values)))
        encoded_lengths = self._downsample_lengths(flat_lengths, self.stem_stride)
        encoded_position = torch.arange(values.shape[2], device=values.device)
        encoded_mask = encoded_position.unsqueeze(0) < encoded_lengths.unsqueeze(1)
        values = values * encoded_mask.unsqueeze(1)
        for block in self.stem_blocks:
            values = block(values) * encoded_mask.unsqueeze(1)
        values = F.gelu(self.output_norm(self.downsample(values)))
        encoded_lengths = self._downsample_lengths(
            encoded_lengths, self.downsample_stride
        )
        encoded_position = torch.arange(values.shape[2], device=values.device)
        mask = encoded_position.unsqueeze(0) < encoded_lengths.unsqueeze(1)
        values = values * mask.unsqueeze(1)
        for block in self.output_blocks:
            values = block(values) * mask.unsqueeze(1)

        mask_values = mask.unsqueeze(1)
        mean = (values * mask_values).sum(dim=2) / encoded_lengths.unsqueeze(1)
        maximum = values.masked_fill(~mask_values, -torch.inf).amax(dim=2)
        attention_logits = self.attention(values).squeeze(1).masked_fill(~mask, -torch.inf)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attended = (values * attention_weights.unsqueeze(1)).sum(dim=2)
        frame_vectors = self.frame_projection(
            torch.cat((mean, maximum, attended), dim=1)
        ).reshape(batch, frames, -1)

        frame_mean = frame_vectors.mean(dim=1)
        frame_maximum = frame_vectors.amax(dim=1)
        original_mask = (
            torch.arange(time, device=tokens.device).view(1, 1, -1)
            < lengths.unsqueeze(2)
        )
        denominator = lengths.to(torch.float32)
        stop_fraction = ((tokens == STOP_ID) & original_mask).sum(dim=2) / denominator
        unknown_fraction = ((tokens == UNKNOWN_ID) & original_mask).sum(dim=2) / denominator
        inferred_bp = 3.0 * lengths.to(torch.float32).amax(dim=1)
        auxiliary = torch.stack(
            (
                torch.log(inferred_bp).div(math.log(100_000.0)),
                stop_fraction.mean(dim=1),
                stop_fraction.std(dim=1, unbiased=False),
                stop_fraction.amax(dim=1),
                unknown_fraction.mean(dim=1),
                unknown_fraction.amax(dim=1),
            ),
            dim=1,
        )
        contig = self.contig_projection(
            torch.cat((frame_mean, frame_maximum, auxiliary), dim=1)
        )
        protein_logit = self.protein_head(contig).squeeze(1)
        residual_logit = self.maximum_residual_logit * torch.tanh(
            self.residual_head(contig).squeeze(1) / self.maximum_residual_logit
        )
        if not torch.isfinite(protein_logit).all() or not torch.isfinite(residual_logit).all():
            raise FloatingPointError("non-finite protein model output")
        return protein_logit, residual_logit


def collate_six_frame_tokens(
    contigs: Sequence[Sequence[np.ndarray]],
) -> tuple[Tensor, Tensor]:
    """Pad a batch of six-frame uint8 arrays for model input."""

    if not contigs or any(len(frames) != 6 for frames in contigs):
        raise ValueError("contigs must be a non-empty sequence of six frames")
    longest = max(len(frame) for frames in contigs for frame in frames)
    if longest < 1:
        raise ValueError("every translated frame must contain at least one token")
    tokens = torch.zeros((len(contigs), 6, longest), dtype=torch.long)
    lengths = torch.empty((len(contigs), 6), dtype=torch.long)
    for contig_index, frames in enumerate(contigs):
        for frame_index, frame in enumerate(frames):
            observed = np.asarray(frame, dtype=np.uint8)
            length = len(observed)
            if length < 1 or np.any(observed == PAD_ID) or np.any(observed >= VOCABULARY_SIZE):
                raise ValueError("translated frames must contain valid non-padding tokens")
            lengths[contig_index, frame_index] = length
            tokens[contig_index, frame_index, :length] = torch.from_numpy(
                observed.astype(np.int64, copy=False)
            )
    return tokens, lengths
