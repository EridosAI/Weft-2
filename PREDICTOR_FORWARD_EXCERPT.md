# Predictor Forward-Pass Excerpt for v1 Design Chat

## File location
/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/src/predictor/inner_pam.py

## Class name
`class InnerPAM(nn.Module)`

## __init__ excerpt (output projection definition)
lines 53-64

```python
        self.input_proj = nn.Linear(embed_dim, hidden)
        self.pos_emb = nn.Embedding(window_w, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden, predict_k * (embed_dim + 1))
```

## forward() excerpt (body output → output projection → reshape)
lines 74-93

```python
        assert window.ndim == 3, f"window must be (B, W, d); got {tuple(window.shape)}"
        b, w, d = window.shape
        assert w == self.window_w, f"W mismatch: got {w}, expected {self.window_w}"
        assert d == self.embed_dim, f"d mismatch: got {d}, expected {self.embed_dim}"

        x = self.input_proj(window)                                 # (B, W, hidden)
        positions = torch.arange(self.window_w, device=window.device)
        x = x + self.pos_emb(positions).unsqueeze(0)                # (B, W, hidden)
        x = self.encoder(x)                                         # (B, W, hidden)
        last_token = x[:, -1, :]                                    # (B, hidden)
        flat = self.output_proj(last_token)                         # (B, K*(d+1))
        flat = flat.view(b, self.predict_k, self.embed_dim + 1)     # (B, K, d+1)
        mean = flat[..., : self.embed_dim]                          # (B, K, d)
        log_var = flat[..., self.embed_dim].clamp(
            LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX
        )                                                            # (B, K)

        assert mean.shape == (b, self.predict_k, self.embed_dim)
        assert log_var.shape == (b, self.predict_k)
        return mean, log_var
```

## Surrounding context (one-paragraph plain-English description)

The transformer body produces a sequence of shape `(B, W, hidden)` where `W` is the window length (W=16) and `hidden` is the transformer hidden dimension (512). On line 83 the forward method selects a single position — the last (most recent) token — via `x[:, -1, :]`, producing one `(B, hidden)` tensor. That single hidden vector is then passed through `self.output_proj` (defined in `__init__` on line 64 as `nn.Linear(hidden, predict_k * (embed_dim + 1))`), producing a flat `(B, K*(d+1))` tensor on line 84. The flat output is reshaped to `(B, K, d+1)` on line 85 via `.view`, then split along the last dimension into a `(B, K, d)` mean and a `(B, K)` log-variance on lines 86-89 (the log-variance is the scalar slice at index `d` of the last dimension, then clamped). There is no learned output-query parameter, no cross-attention read-out, and no per-position output head: the only positional structure in the module is `self.pos_emb` (line 54), which is added to the input embeddings before the encoder body (line 81), not consumed as output queries. K is produced entirely by the shape of `self.output_proj`'s output, fanning out from one pooled hidden vector.

## Filenames and line numbers, for reference
/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/src/predictor/inner_pam.py:53-64
/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/src/predictor/inner_pam.py:74-93
