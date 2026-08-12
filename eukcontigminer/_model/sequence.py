"""Reverse-complement-invariant 1D CNN for short nuclear Euk detection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

PAD_TOKEN = 5


@dataclass(frozen=True)
class SequenceCNNConfig:
    vocabulary_size: int = 6
    pad_token: int = PAD_TOKEN
    embedding_dimension: int = 16
    channels: int = 96
    kernel_size: int = 7
    stem_kernel_size: int = 15
    dilations: tuple[int, ...] = (1, 3, 9, 27)
    hidden_dimension: int = 256
    dropout: float = 0.1
    normalization: str = "group"
    maximum_length: int = 2500
    output_classes: int = 2

    def validate(self) -> None:
        if self.vocabulary_size <= 0:
            raise ValueError("vocabulary size must be positive")
        if not 0 <= self.pad_token < self.vocabulary_size:
            raise ValueError("padding token is outside vocabulary")
        for value in (
            self.embedding_dimension,
            self.channels,
            self.hidden_dimension,
            self.maximum_length,
            self.output_classes,
        ):
            if value <= 0:
                raise ValueError("model dimensions must be positive")
        for kernel in (self.kernel_size, self.stem_kernel_size):
            if kernel <= 0 or kernel % 2 == 0:
                raise ValueError("convolution kernels must be positive odd")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.normalization not in {"group", "masked_group"}:
            raise ValueError(
                "normalization must be group or masked_group"
            )


class MaskedGroupNorm1d(nn.Module):
    """Group normalization over channels and valid sequence positions only."""

    def __init__(
        self,
        channels: int,
        *,
        groups: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups:
            raise ValueError("invalid masked GroupNorm dimensions")
        self.channels = channels
        self.groups = groups
        self.channels_per_group = channels // groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 3 or mask.shape != (
            len(values),
            1,
            values.shape[2],
        ):
            raise ValueError("masked GroupNorm inputs have invalid shapes")
        source_dtype = values.dtype
        source = (
            values.float()
            if source_dtype in {torch.float16, torch.bfloat16}
            else values
        )
        valid = mask.to(dtype=source.dtype)
        grouped = source.reshape(
            len(source),
            self.groups,
            self.channels_per_group,
            source.shape[2],
        )
        group_mask = valid.unsqueeze(1)
        count = (
            valid.sum(dim=2, keepdim=False)
            * self.channels_per_group
        ).clamp_min(1)
        mean = (grouped * group_mask).sum(dim=(2, 3)) / count
        centered = grouped - mean.unsqueeze(2).unsqueeze(3)
        variance = (
            centered.square() * group_mask
        ).sum(dim=(2, 3)) / count
        normalized = centered * torch.rsqrt(
            variance.unsqueeze(2).unsqueeze(3) + self.eps
        )
        normalized = normalized.reshape_as(source).to(source_dtype)
        weight = self.weight.to(source_dtype).view(1, -1, 1)
        bias = self.bias.to(source_dtype).view(1, -1, 1)
        return (normalized * weight + bias) * mask


def _group_count(channels: int) -> int:
    groups = min(16, channels)
    while channels % groups:
        groups -= 1
    return groups


def _sequence_normalization(
    channels: int,
    normalization: str,
) -> nn.Module:
    groups = _group_count(channels)
    if normalization == "group":
        return nn.GroupNorm(groups, channels)
    if normalization == "masked_group":
        return MaskedGroupNorm1d(channels, groups=groups)
    raise ValueError(f"unsupported sequence normalization {normalization}")


def _apply_sequence_normalization(
    normalization: nn.Module,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if isinstance(normalization, MaskedGroupNorm1d):
        return normalization(values, mask)
    return normalization(values)


class ResidualSequenceBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        normalization: str,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.normalization = _sequence_normalization(
            channels,
            normalization,
        )
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise_in = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=1,
        )
        self.pointwise_out = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = values
        hidden = _apply_sequence_normalization(
            self.normalization,
            values,
            mask,
        )
        hidden = self.depthwise(hidden)
        hidden = nn.functional.glu(self.pointwise_in(hidden), dim=1)
        hidden = self.dropout(self.pointwise_out(hidden))
        return (residual + hidden) * mask
class ReverseComplementSequenceCNN(nn.Module):
    """Padding-aware sequence classifier; RC averaging lives in predictor."""

    def __init__(self, config: SequenceCNNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocabulary_size,
            config.embedding_dimension,
            padding_idx=config.pad_token,
        )
        self.stem = nn.Conv1d(
            config.embedding_dimension,
            config.channels,
            config.stem_kernel_size,
            padding=(config.stem_kernel_size - 1) // 2,
        )
        self.blocks = nn.ModuleList(
            ResidualSequenceBlock(
                config.channels,
                kernel_size=config.kernel_size,
                dilation=dilation,
                dropout=config.dropout,
                normalization=config.normalization,
            )
            for dilation in config.dilations
        )
        self.final_normalization = _sequence_normalization(
            config.channels,
            config.normalization,
        )
        self.classifier = nn.Sequential(
            nn.Linear(
                config.channels * 2 + 1,
                config.hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dimension, config.output_classes),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or lengths.shape != (len(tokens),):
            raise ValueError("tokens and lengths have incompatible shapes")
        if tokens.shape[1] > self.config.maximum_length:
            raise ValueError("input exceeds configured maximum length")
        if torch.any(lengths <= 0) or torch.any(
            lengths > tokens.shape[1]
        ):
            raise ValueError("sequence lengths are invalid")
        positions = torch.arange(
            tokens.shape[1],
            device=tokens.device,
        ).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        mask = valid.unsqueeze(1).to(dtype=torch.float32)
        masked_tokens = torch.where(
            valid,
            tokens,
            torch.full_like(tokens, self.config.pad_token),
        )
        hidden = self.embedding(masked_tokens).transpose(1, 2)
        hidden = nn.functional.gelu(self.stem(hidden)) * mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = _apply_sequence_normalization(
            self.final_normalization,
            hidden,
            mask,
        ) * mask
        denominator = lengths.to(hidden.dtype).clamp_min(1).unsqueeze(1)
        mean = hidden.sum(dim=2) / denominator
        negative_infinity = torch.finfo(hidden.dtype).min
        maximum = hidden.masked_fill(
            ~valid.unsqueeze(1),
            negative_infinity,
        ).amax(dim=2)
        normalized_log_length = (
            torch.log(lengths.to(hidden.dtype))
            / np.log(float(self.config.maximum_length))
        ).unsqueeze(1)
        pooled = torch.cat(
            (mean, maximum, normalized_log_length),
            dim=1,
        )
        return self.classifier(pooled)


def reverse_complement_batch(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    pad_token: int = PAD_TOKEN,
) -> torch.Tensor:
    """Reverse-complement a padded token batch on its current device."""
    if tokens.ndim != 2 or lengths.shape != (len(tokens),):
        raise ValueError("tokens and lengths have incompatible shapes")
    width = tokens.shape[1]
    if torch.any(lengths <= 0) or torch.any(lengths > width):
        raise ValueError("sequence lengths are invalid")
    positions = torch.arange(width, device=tokens.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    source = lengths.unsqueeze(1) - 1 - positions
    source = source.clamp(min=0, max=max(0, width - 1))
    reversed_tokens = tokens.gather(1, source)
    complement = torch.tensor(
        [3, 2, 1, 0, 4, pad_token],
        dtype=torch.long,
        device=tokens.device,
    )
    complemented = complement[reversed_tokens.long()]
    return torch.where(
        valid,
        complemented.to(tokens.dtype),
        torch.full_like(tokens, pad_token),
    )


class SequenceCNNPredictor:
    def __init__(
        self,
        model: ReverseComplementSequenceCNN,
        *,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(
        self,
        tokens: np.ndarray | torch.Tensor,
        lengths: np.ndarray | torch.Tensor,
        *,
        batch_size: int = 512,
        reverse_complement_average: bool = True,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        token_array = (
            tokens.detach().cpu().numpy()
            if isinstance(tokens, torch.Tensor)
            else np.asarray(tokens)
        )
        length_array = (
            lengths.detach().cpu().numpy()
            if isinstance(lengths, torch.Tensor)
            else np.asarray(lengths)
        )
        if token_array.ndim != 2 or length_array.shape != (
            len(token_array),
        ):
            raise ValueError("tokens and lengths have incompatible shapes")
        outputs = []
        use_bfloat16 = (
            self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        for start in range(0, len(token_array), batch_size):
            stop = min(len(token_array), start + batch_size)
            batch_lengths = torch.as_tensor(
                length_array[start:stop],
                dtype=torch.long,
                device=self.device,
            )
            width = int(batch_lengths.max().item())
            batch_tokens = torch.as_tensor(
                token_array[start:stop, :width],
                dtype=torch.long,
                device=self.device,
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                probabilities = torch.softmax(
                    self.model(batch_tokens, batch_lengths),
                    dim=1,
                )
                if reverse_complement_average:
                    reverse = reverse_complement_batch(
                        batch_tokens,
                        batch_lengths,
                        pad_token=self.model.config.pad_token,
                    )
                    probabilities = 0.5 * (
                        probabilities
                        + torch.softmax(
                            self.model(reverse, batch_lengths),
                            dim=1,
                        )
                    )
            outputs.append(
                probabilities.to(torch.float32).cpu().numpy()
            )
        return np.concatenate(outputs, axis=0)


def eukaryota_probability(probabilities: np.ndarray) -> np.ndarray:
    """Collapse supported CNN outputs to a nuclear-Euk probability."""
    values = np.asarray(probabilities)
    if values.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    tolerance = 1e-6
    if np.any(~np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    if np.any((values < -tolerance) | (values > 1 + tolerance)):
        raise ValueError("probabilities must be in [0,1]")
    if values.shape[1] == 2:
        scores = np.asarray(values[:, 1], dtype=np.float64)
    elif values.shape[1] == 6:
        scores = np.asarray(values[:, 2:], dtype=np.float64).sum(axis=1)
    else:
        raise ValueError(
            "nuclear-Euk scoring requires either 2 root classes or "
            "6 detail classes"
        )
    if np.any((scores < -tolerance) | (scores > 1 + tolerance)):
        raise ValueError("collapsed nuclear-Euk scores must be in [0,1]")
    return np.clip(scores, 0.0, 1.0)


def save_sequence_cnn(
    model: ReverseComplementSequenceCNN,
    directory: str | Path,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "model.json").write_text(
        json.dumps(
            {
                "schema": "reverse_complement_sequence_cnn_v1",
                "config": asdict(model.config),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(model.state_dict(), directory / "weights.pt")


def load_sequence_cnn(
    directory: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ReverseComplementSequenceCNN:
    directory = Path(directory)
    payload = json.loads(
        (directory / "model.json").read_text(encoding="utf-8")
    )
    if payload.get("schema") != "reverse_complement_sequence_cnn_v1":
        raise ValueError("unsupported sequence CNN model schema")
    raw_config = dict(payload["config"])
    raw_config["dilations"] = tuple(raw_config["dilations"])
    config = SequenceCNNConfig(**raw_config)
    model = ReverseComplementSequenceCNN(config)
    state = torch.load(
        directory / "weights.pt",
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state)
    return model
