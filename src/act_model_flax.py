# src/act_model_flax.py
import jax, jax.numpy as jnp
import flax.linen as nn

# ------------------------------------------------------------
# Shared dimensions
# ------------------------------------------------------------
D_EMB = 256   # Visual/State embedding size
Z_DIM = 32    # Latent size (CVAE)

# ============================================================
#   ENCODER OPTIONS
#   - ImgEncViTB16 : Pretrained ViT-B/16 (closest to original)
#   - ImgEncViT    : MiniViT (scratch) – no internet required
#   - ImgEncCNN    : Lightweight CNN – baseline
# ============================================================

# ---------------- (Optional) ViT-B/16 Pretrained ----------------
# This encoder internally upsamples 64x64 to 224x224 (dataset unchanged).
try:
    from transformers import FlaxViTModel
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

class ImgEncViTB16(nn.Module):
    """Upsamples three 64x64 camera views (worm/side/wrist) to 224 and
    encodes them with HF FlaxViT-B/16; projects output to D_EMB.
    """
    freeze_backbone: bool = True  # True: do not send grads to ViT backbone

    @nn.compact
    def __call__(self, x):  # x: [B,T,64,64,9]
        if not _HAS_TRANSFORMERS:
            raise RuntimeError(
                "enc_type='vit_b16' was selected but the 'transformers' package was not found. "
                "Install with: uv pip install 'transformers>=4.44.0'"
            )
        B, T, H, W, C9 = x.shape
        assert C9 == 9, "ViT-B/16 encoder expects 3 cameras (9 channels: 3xRGB)."

        worm, side, wrist = x[..., 0:3], x[..., 3:6], x[..., 6:9]

        def bt(v): return v.reshape(B * T, H, W, 3)  # [B*T,H,W,3]
        def resize224(v):
            # NHWC -> NHWC, 64->224
            return jax.image.resize(v, (v.shape[0], 224, 224, 3), method="linear")
        def nhwc_to_nchw(v):
            return jnp.transpose(v, (0, 3, 1, 2))  # [N,3,224,224]

        vit = FlaxViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            dtype=jnp.float32,
            add_pooling_layer=False,  # returns last_hidden_state
        )

        outs = []
        for img in (worm, side, wrist):
            v = bt(img)           # [B*T,64,64,3]
            v = resize224(v)      # [B*T,224,224,3]
            pv = nhwc_to_nchw(v)  # [B*T,3,224,224]
            hs = vit(pixel_values=pv).last_hidden_state  # [B*T, N, 768]
            if self.freeze_backbone:
                hs = jax.lax.stop_gradient(hs)  # stop grads to ViT
            feat = hs.mean(axis=1)  # token mean -> [B*T,768]
            outs.append(feat)

        emb_bt = sum(outs) / 3.0               # [B*T,768]
        emb_bt = nn.Dense(D_EMB)(emb_bt)       # [B*T,D_EMB]
        return emb_bt.reshape(B, T, D_EMB)     # [B,T,D_EMB]

# ---------------- MiniViT (scratch) ----------------
class PatchEmbed(nn.Module):
    patch: int
    dim: int
    @nn.compact
    def __call__(self, x):  # [B, H, W, 3]
        y = nn.Conv(self.dim, (self.patch, self.patch), self.patch, use_bias=False)(x)  # [B,H/ps,W/ps,dim]
        B, Ht, Wt, C = y.shape
        y = y.reshape(B, Ht * Wt, C)  # [B, N, dim]
        return y, (Ht, Wt)


class MLP(nn.Module):
    dim: int
    mlp_ratio: float = 4.0
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(int(self.dim * self.mlp_ratio))(x)
        h = nn.gelu(h)
        h = nn.Dense(self.dim)(h)
        return h

class EncoderBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.dim, out_features=self.dim)(h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = MLP(self.dim, self.mlp_ratio)(h)
        x = x + h
        return x

class MiniViT(nn.Module):
    dim: int = D_EMB
    depth: int = 6
    num_heads: int = 8
    patch: int = 16  # 64x64 -> 4*4=16 tokens
    @nn.compact
    def __call__(self, x):  # [B, H, W, 3]
        tokens, _ = PatchEmbed(self.patch, self.dim)(x)  # [B, N, D]
        B, N, D = tokens.shape
        pos = self.param("pos_embed", nn.initializers.normal(0.02), (1, N, D))
        h = tokens + pos
        for _ in range(self.depth):
            h = EncoderBlock(self.dim, self.num_heads)(h)
        return h.mean(axis=1)  # [B, D]

class ImgEncViT(nn.Module):
    """Encodes three cameras with a shared MiniViT and averages them."""
    @nn.compact
    def __call__(self, x):  # [B,T,64,64,9]
        B, T, H, W, C9 = x.shape
        assert C9 == 9, "MiniViT encoder expects 3 cameras (9 channels)."
        worm, side, wrist = x[..., 0:3], x[..., 3:6], x[..., 6:9]
        def bt(v): return v.reshape(B * T, H, W, 3)
        vit = MiniViT()   # shared
        e1 = vit(bt(worm));  e2 = vit(bt(side));  e3 = vit(bt(wrist))  # [B*T,D]
        emb_bt = (e1 + e2 + e3) / 3.0
        return emb_bt.reshape(B, T, D_EMB)  # [B,T,D]

# ---------------- Lightweight CNN ----------------
class ImgEncCNN(nn.Module):
    @nn.compact
    def __call__(self, x):  # [B,T,64,64,9]
        B, T, H, W, C = x.shape
        y = x.reshape(B * T, H, W, C)
        y = nn.Conv(32, (5, 5), 2)(y); y = nn.relu(y)
        y = nn.Conv(64, (3, 3), 2)(y); y = nn.relu(y)
        y = nn.Conv(128,(3, 3), 2)(y); y = nn.relu(y)
        y = y.mean(axis=(1, 2))
        y = nn.Dense(D_EMB)(y)
        return y.reshape(B, T, D_EMB)

# ============================================================
#   STATE ENCODER
# ============================================================
class StateEnc(nn.Module):
    @nn.compact
    def __call__(self, joints, gripper):  # [B,T,J], [B,T,1]
        x = jnp.concatenate([joints, gripper], -1)
        x = nn.Dense(128)(x); x = nn.relu(x)
        x = nn.Dense(D_EMB)(x)
        return x  # [B,T,D]

# ============================================================
#   VAE HEADS (Prior / Posterior)
# ============================================================
class Heads(nn.Module):
    out_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x); x = nn.tanh(x)
        return nn.Dense(self.out_dim)(x)

# ============================================================
#   CAUSAL DECODER
# ============================================================
class CausalDecoder(nn.Module):
    out_dim: int  # A
    length: int   # H
    @nn.compact
    def __call__(self, ctx, z):  # ctx=[B,Tctx,D], z=[B,Z]
        B = ctx.shape[0]
        zt   = jnp.tile(z[:, None, :], [1, self.length, 1])                  # [B,H,Z]
        ctxm = jnp.mean(ctx, axis=1, keepdims=True).repeat(self.length, 1)   # [B,H,D]
        x = jnp.concatenate([ctxm, zt], -1)                                  # [B,H,D+Z]
        x = nn.Dense(256)(x); x = nn.relu(x)
        x = nn.SelfAttention(num_heads=4, qkv_features=256)(x)
        return nn.Dense(self.out_dim)(x)                                     # [B,H,A]

# ============================================================
#   ACT MODEL (Close to the original)
# ============================================================
class ACTModel(nn.Module):
    action_dim: int            # A
    chunk_len: int             # H
    enc_type: str = "vit"      # "vit_b16" | "vit" | "cnn"
    freeze_backbone: bool = True  # for vit_b16


    @nn.compact
    def __call__(self, images, joints, gripper, actions=None, train: bool = True, beta: float = 1e-3):
        # ---- Visual encoder selection ----
        if self.enc_type == "vit_b16":
            x_img = ImgEncViTB16(freeze_backbone=self.freeze_backbone)(images)
        elif self.enc_type == "vit":
            x_img = ImgEncViT()(images)
        else:
            x_img = ImgEncCNN()(images)

        # ---- State ----
        x_sta = StateEnc()(joints, gripper)

        # ---- Context ----
        ctx  = x_img + x_sta                # [B,T,D]
        ctxm = jnp.mean(ctx, axis=1)        # [B,D]

        # ---- Prior p(z|O) ----
        prior = Heads(2 * Z_DIM)(ctxm)
        mu_p, ls_p = jnp.split(prior, 2, -1)
        std_p = jnp.exp(ls_p)

        # ---- Posterior q(z|O,A) (training only) ----
        if train and actions is not None:
            tgt = actions.reshape(actions.shape[0], -1)  # [B,H*A]
            post = Heads(2 * Z_DIM)(jnp.concatenate([ctxm, tgt], -1))
            mu_q, ls_q = jnp.split(post, 2, -1)
            std_q = jnp.exp(ls_q)
            eps = jax.random.normal(self.make_rng("noise"), mu_q.shape)
            z = mu_q + std_q * eps
            # KL(q||p) as a scalar per-sample
            kl = jnp.sum((ls_p - ls_q) + (std_q**2 + (mu_q - mu_p)**2) / (2 * std_p**2) - 0.5, axis=-1)  # [B]
        else:
            z = mu_p
            kl = jnp.zeros((images.shape[0],), images.dtype)

        # ---- Decoder ----
        pred = CausalDecoder(self.action_dim, self.chunk_len)(ctx, z)  # [B,H,A]
        return pred, kl
