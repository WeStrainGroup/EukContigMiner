"""Same NTv3 weights/windows/FP32; faster CPU encoding and homogeneous batches."""
import numpy as np
import torch


def install(batch_windows=16):
    from eukcontigminer import ntv3_runtime as nt
    assert 1 <= batch_windows <= 64
    if getattr(nt.NTV3Adapter, '_ecm_fast_batch_windows', None) == batch_windows:
        return
    nt.NTV3Adapter._ecm_fast_batch_windows = batch_windows
    original_encode = nt.NTV3Adapter.encode

    def encode(self, seqs):
        if not hasattr(self, '_ecm_ascii_ids'):
            mapping = np.full(256, -1, dtype=np.int64)
            for char in 'ACGTN':
                value = self.tokenizer(char, add_special_tokens=False)['input_ids']
                assert len(value) == 1
                mapping[ord(char)] = value[0]
            # Verify actual pinned tokenizer on combinations before replacing it.
            for s in ['ACGTN', 'NNN', 'GATTACA', 'ACGT'*33]:
                assert np.array_equal(mapping[np.frombuffer(s.encode(), dtype=np.uint8)],
                                      self.tokenizer(s, add_special_tokens=False)['input_ids'])
            self._ecm_ascii_ids = mapping
        width = ((max(map(len, seqs)) + 127) // 128) * 128
        a = np.full((len(seqs), width), self._ecm_ascii_ids[ord('N')], np.int64)
        mask = np.zeros(a.shape, np.int64)
        for i, s in enumerate(seqs):
            ids = self._ecm_ascii_ids[np.frombuffer(s.encode('ascii'), dtype=np.uint8)]
            if (ids < 0).any():
                return original_encode(self, seqs)
            a[i, :len(s)] = ids
            mask[i, :len(s)] = 1
        assert not (a == self.tokenizer.pad_token_id).any()
        return torch.from_numpy(a).to(self.device), torch.from_numpy(mask).to(self.device)

    def predict_windows(self, sequences):
        sequences = [nt.normalize(s) for s in sequences]
        buckets = {}
        for i, s in enumerate(sequences):
            if len(s) > 2000:
                raise ValueError('NTv3 window exceeds 2000 bp')
            buckets.setdefault((len(s)+127)//128, []).append(i)
        result = np.full(len(sequences), np.nan)
        results, positions = [], []
        with torch.no_grad(), nt.ieee_fp32(self.device):
            for width, indices in sorted(buckets.items()):
                for start in range(0, len(indices), batch_windows):
                    take = indices[start:start+batch_windows]
                    views = []
                    for i in take:
                        views.extend(sorted([sequences[i], sequences[i].translate(nt.RC)[::-1]]))
                    ids, mask = self.encode(views)
                    assert ids.shape[1] == width*128
                    # The caller already guarantees a homogeneous padding bucket.
                    output = self.model.encoder.core(input_ids=ids)
                    h = output['embeddings_deconv_'+str(self.model.encoder.config.num_downsamples)].permute(0,2,1).float()
                    valid = mask.bool().unsqueeze(-1)
                    pooled = torch.cat([(h*valid).sum(1)/valid.sum(1),
                                        h.masked_fill(~valid, -torch.inf).amax(1)], 1)
                    z = self.model.head(pooled).squeeze(-1).float()
                    results.append(z.double().reshape(-1,2).mean(1))
                    positions.extend(take)
            if results:
                result[np.asarray(positions)] = torch.cat(results).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('NTv3 emitted nonfinite logits')
        self.windows += len(sequences)
        return result

    nt.NTV3Adapter.encode = encode
    nt.NTV3Adapter.predict_windows = predict_windows
