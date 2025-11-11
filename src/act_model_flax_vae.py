# act_model_flax_vae.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
from resnet_encoder_flax import ResNetEncoder

class TransformerBlock(nn.Module):
    d_model: int; n_heads: int; mlp_ratio: float = 4.0; dropout: float = 0.0
    @nn.compact
    def __call__(self, x, *, deterministic: bool, attn_mask: Optional[jnp.ndarray]):
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            use_bias=True,
            broadcast_dropout=False
        )(h, mask=attn_mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(int(self.d_model*self.mlp_ratio))(h); h = nn.gelu(h)
        h = nn.Dropout(self.dropout)(h, deterministic=deterministic)
        h = nn.Dense(self.d_model)(h)
        h = nn.Dropout(self.dropout)(h, deterministic=deterministic)
        return x + h

class TransformerDecoder(nn.Module):
    d_model: int; depth: int; n_heads: int; dropout: float = 0.0
    @nn.compact
    def __call__(self, x, *, deterministic: bool, attn_mask: Optional[jnp.ndarray]):
        for _ in range(self.depth):
            x = TransformerBlock(self.d_model, self.n_heads, dropout=self.dropout)(
                x, deterministic=deterministic, attn_mask=attn_mask)
        return nn.LayerNorm()(x)

class ACTVAEModel(nn.Module):
    """
    Simplified cVAE close to ACT:
      - Encoder: (actions_chunk, qpos) -> (mu, logvar)
      - Latent:  z = mu + exp(0.5*logvar) * eps
      - Decoder: [img_ctx, proprio_ctx, z token, action_queries] -> actions_hat
    Output: (actions_pred[B,H,A], kl_per_example[B])
    """

    action_dim: int          # 8
    chunk_len: int           # H
    d_model: int = 256
    depth: int = 8
    n_heads: int = 8
    dropout: float = 0.0
    resnet_variant: str = "resnet18"
    proprio_dim: int = 8     # joints(7)+gripper(1)
    latent_dim: int = 64

    # -------- utilities --------
    def _build_ctx(self, images, joints, gripper, *, train: bool):
        B, Tctx, H, W, C9 = images.shape
        assert C9 == 9,"Expecting 3 cameras: low/side/wrist -> 9 channels."
        low   = images[..., :3].reshape(B*Tctx, H, W, 3)
        side  = images[..., 3:6].reshape(B*Tctx, H, W, 3)
        wrist = images[..., 6:9].reshape(B*Tctx, H, W, 3)

        enc = ResNetEncoder(variant=self.resnet_variant, d_model=self.d_model, name="shared_resnet")
        e_low   = enc(low,   train=train)
        e_side  = enc(side,  train=train)
        e_wrist = enc(wrist, train=train)
        img_ctx = (e_low + e_side + e_wrist) / 3.0                   # [B*Tctx, D]
        img_ctx = img_ctx.reshape(B, Tctx, self.d_model)              # [B, Tctx, D]

        prop = jnp.concatenate([joints, gripper], axis=-1)            # [B,Tctx,8]
        prop = nn.Dense(256)(prop); prop = nn.gelu(prop)
        prop = nn.Dense(self.d_model)(prop)                           # [B,Tctx,D]
        return img_ctx, prop

    def _posterior(self, actions: jnp.ndarray, qpos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # actions embed
        h = nn.Dense(self.d_model)(actions)            # [B,H,D]
        h = nn.gelu(h)
        h = jnp.mean(h, axis=1)                        # [B,D]
        # qpos proj
        q = nn.Dense(self.d_model)(qpos)               # [B,D]
        h = h + q
        mu     = nn.Dense(self.latent_dim, name="post_mu")(h)         # [B,Z]
        logvar = nn.Dense(self.latent_dim, name="post_logvar")(h)     # [B,Z]
        return mu, logvar

    @nn.compact
    def __call__(self,
                 images: jnp.ndarray,     # [B,Tctx,H,W,9]
                 joints: jnp.ndarray,     # [B,Tctx,7]
                 gripper: jnp.ndarray,    # [B,Tctx,1]
                 *,
                 actions_chunk: Optional[jnp.ndarray] = None,  # [B,H,8] (train’de ver)
                 train: bool = True):
        B, Tctx, *_ = images.shape

        # ---- context: resnet + proprio ----
        img_ctx, prop_ctx = self._build_ctx(images, joints, gripper, train=train)   # [B,Tctx,D] x 2
        ctx = jnp.concatenate([img_ctx, prop_ctx], axis=1)                          # [B,2*Tctx,D]

        # qpos
        qpos = jnp.concatenate([joints[:, -1, :], gripper[:, -1, :]], axis=-1)      # [B,8]

        # ---- latent (posterior or prior) ----
        if (actions_chunk is not None) and train:
            mu, logvar = self._posterior(actions_chunk, qpos)                       # [B,Z] x2
            eps = jax.random.normal(self.make_rng("latent"), mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps                                    # [B,Z]
            # KL per example: 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
            kl = 0.5 * jnp.sum(jnp.exp(logvar) + mu**2 - 1.0 - logvar, axis=-1)     # [B]
        else:
            # prior (in inference or when actions are not available): batch-aligned zeros
            mu = jnp.zeros((B, self.latent_dim), dtype=jnp.float32)
            logvar = jnp.zeros_like(mu)
            z = jnp.zeros_like(mu)
            kl = jnp.zeros((B,), dtype=jnp.float32)

        z_tok = nn.Dense(self.d_model, name="z_proj")(z)                             # [B,D]
        z_tok = z_tok[:, None, :]                                                   # [B,1,D]

        # ---- action queries + pos ----
        Hchunk = self.chunk_len
        action_queries = self.param("action_queries", nn.initializers.normal(0.02),
                                    (Hchunk, self.d_model))
        action_queries = jnp.broadcast_to(action_queries[None, ...], (B, Hchunk, self.d_model))

        x = jnp.concatenate([ctx, z_tok, action_queries], axis=1)                   # [B,2*Tctx+1+H, D]
        T = x.shape[1]
        pos = self.param("pos_emb", nn.initializers.normal(0.02), (T, self.d_model))
        x = x + pos[None, :, :]

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn_mask = causal[None, None, :, :]

        x = TransformerDecoder(self.d_model, self.depth, self.n_heads, dropout=self.dropout)(
            x, deterministic=not train, attn_mask=attn_mask
        )

        act_h = x[:, -Hchunk:, :]                                                   # [B,H,D]
        actions_pred = nn.Dense(self.action_dim, name="action_head")(act_h)         # [B,H,A]

        return actions_pred, kl
