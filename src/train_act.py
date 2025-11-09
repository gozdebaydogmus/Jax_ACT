# train_act.py
import os, pathlib
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
from jax import tree_util

from dataloader_jax import make_batches

# --------- Model selection: VAE (latent+KL) > DET fallback ---------
USE_VAE = True
MODEL = None
optimizer = None

try:
    if USE_VAE:
        from act_model_flax_vae import ACTVAEModel as ACTModel
    else:
        raise ImportError("Force DET")
except Exception:
    from act_model_flax_det import ACTDeterministic as ACTModel
    USE_VAE = False

# ---------------- Env setup ----------------
os.environ.setdefault("TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets"))
os.environ.setdefault("TFDS_WORK_DIR", os.path.expanduser("~/.cache/tfds_work"))
os.makedirs(os.environ["TFDS_WORK_DIR"], exist_ok=True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

CKPT_DIR = pathlib.Path("logs/ckpt").expanduser().resolve()
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Optimization & beta ----------------
def make_optimizer():
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=1_000,
        decay_steps=50_000,
        end_value=1e-4,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(sched, weight_decay=1e-4),
    )

def beta_anneal(step, warmup=1_000, max_beta=1e-3):
    s = float(step)
    frac = min(1.0, s / float(warmup))
    return max_beta * frac


# --------- COMPATIBILITY: (pred, kl) interface ---------
def model_apply_compat(params, batch_stats, images, joints, gripper, *, actions_chunk=None, train: bool):
    """
    Expected model outputs:
      - VAE version: returns (pred[B,H,A], kl_per_example[B])
      - DET version: returns pred[B,H,A] -> here we generate KL=0 vector
    If BatchNorm exists, 'batch_stats' is mutable/updated.
    """


    variables = {"params": params}
    if batch_stats is not None:
        variables["batch_stats"] = batch_stats

    rngs = {"dropout": jax.random.PRNGKey(0), "latent": jax.random.PRNGKey(1)} if train else None
    mutable = ["batch_stats"] if (train and batch_stats is not None) else False

    out = MODEL.apply(
        variables,
        images, joints, gripper,
        actions_chunk=actions_chunk if train else None,
        train=train,
        rngs=rngs,
        mutable=mutable
    )

    if train and mutable:
        (model_out, new_vars) = out
        new_batch_stats = new_vars.get("batch_stats", batch_stats)
    else:
        model_out = out
        new_batch_stats = batch_stats

    if isinstance(model_out, (tuple, list)) and len(model_out) == 2:
        pred, kl = model_out
    else:
        pred = model_out
        kl = jnp.zeros((pred.shape[0],), dtype=pred.dtype)  # DET fallback or secure

    pred = jnp.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    kl   = jnp.nan_to_num(kl,   nan=0.0, posinf=1e6, neginf=0.0)

    return pred, kl, new_batch_stats

# ---------------- PMAP steps ----------------
@partial(jax.pmap, axis_name="dev")
def train_step(params, batch_stats, opt_state, batch, beta_scalar):
    images  = jnp.asarray(batch["images"], dtype=jnp.float32)
    joints  = jnp.asarray(batch["joints"], dtype=jnp.float32)
    gripper = jnp.asarray(batch["gripper"], dtype=jnp.float32)
    target  = jnp.asarray(batch["target_actions"], dtype=jnp.float32)

    def loss_fn(p, bs):
        pred, kl, new_bs = model_apply_compat(
            p, bs, images, joints, gripper,
            actions_chunk=target,  # For VAE posterior
            train=True
        )

        # --- L1 ---
        rec = jnp.mean(jnp.abs(pred - target), axis=(1, 2))  # (B)
        rec_mean = jnp.mean(rec)
        kl_mean = jnp.mean(jnp.clip(kl, 0.0, 1e6))

        loss = rec_mean + beta_scalar * kl_mean

        rec_ok = jnp.all(jnp.isfinite(rec))
        kl_ok  = jnp.all(jnp.isfinite(kl))
        metrics = {"rec": rec_mean, "kl": kl_mean, "rec_ok": rec_ok, "kl_ok": kl_ok}
        return loss, (metrics, new_bs)

    (loss, (metrics, new_bs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch_stats)

    loss    = jax.lax.pmean(loss,    axis_name="dev")
    metrics = jax.lax.pmean(metrics, axis_name="dev")
    grads   = jax.lax.pmean(grads,   axis_name="dev")

    if isinstance(new_bs, dict) or (hasattr(new_bs, "keys") and len(new_bs) > 0):
        new_bs = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="dev"), new_bs)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, new_bs, opt_state, {"loss": loss, **metrics}

@partial(jax.pmap, axis_name="dev")
def eval_step(params, batch_stats, batch):
    images  = jnp.asarray(batch["images"], dtype=jnp.float32)
    joints  = jnp.asarray(batch["joints"], dtype=jnp.float32)
    gripper = jnp.asarray(batch["gripper"], dtype=jnp.float32)
    target  = jnp.asarray(batch["target_actions"], dtype=jnp.float32)

    pred, _, _ = model_apply_compat(params, batch_stats, images, joints, gripper, train=False)
    mae = jnp.mean(jnp.abs(pred - target))  
    return jax.lax.pmean(mae, axis_name="dev")

# -------------------- Main script --------------------
def main():
    global MODEL, optimizer

    # 1) First batch (init & shape reference)
    batch_size = 4
    assert batch_size % jax.local_device_count() == 0, "batch_size, cihaz sayısına bölünmeli."
    it = make_batches(batch_size=batch_size)
    first = next(it)

    act_dim   = first["target_actions"].shape[-1]  # 8
    chunk_len = first["target_actions"].shape[2]   # H

    # 2) Model
    if USE_VAE:
        MODEL = ACTModel(
            action_dim=act_dim,
            chunk_len=chunk_len,
            d_model=256,
            depth=8,
            n_heads=8,
            dropout=0.0,
            resnet_variant="resnet18",
            proprio_dim=8,
            latent_dim=128,
        )
    else:
        MODEL = ACTModel(
            action_dim=act_dim,
            chunk_len=chunk_len,
            d_model=256,
            depth=8,
            n_heads=8,
            dropout=0.0,
            resnet_variant="resnet18",
            proprio_dim=8,
        )

    # 3) Init using single-device slice
    single = {k: val[0] for k, val in first.items()}
    variables = MODEL.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0), "latent": jax.random.PRNGKey(1)},
        single["images"], single["joints"], single["gripper"],
        actions_chunk=single["target_actions"],
        train=True,
    )
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    # Load checkpoint (optional)
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=str(CKPT_DIR),
        target={"params": params, "batch_stats": batch_stats}
    )
    params      = restored.get("params", params)
    batch_stats = restored.get("batch_stats", batch_stats)

    # 4) Optimization & replicate
    optimizer_local = make_optimizer()
    opt_state = optimizer_local.init(params)
    params      = jax.device_put_replicated(params, jax.local_devices())
    if isinstance(batch_stats, dict) and len(batch_stats) > 0:
        batch_stats = jax.device_put_replicated(batch_stats, jax.local_devices())
    else:
        batch_stats = jax.device_put_replicated({}, jax.local_devices())
    opt_state   = jax.device_put_replicated(opt_state, jax.local_devices())
    globals()["optimizer"] = optimizer_local

    # 5) Training
    total_steps = 100000
    log_every   = 5

    for step in range(total_steps):
        beta = beta_anneal(step, warmup=1000, max_beta=1e-3) if USE_VAE else 0.0
        beta_vec = jnp.full((jax.local_device_count(),), beta, dtype=jnp.float32)

        batch = first if step == 0 else next(it)
        params, batch_stats, opt_state, metrics = train_step(params, batch_stats, opt_state, batch, beta_vec)

        if (step % log_every == 0) and (jax.process_index() == 0):
            m = jax.device_get(metrics)
            loss = float(m["loss"][0]); rec  = float(m["rec"][0]); kl = float(m["kl"][0])
            rec_ok = bool(m["rec_ok"][0]); kl_ok  = bool(m["kl_ok"][0])
            print(f"step {step:5d}  beta={float(beta):.6g}  loss={loss:.6f}  rec={rec:.6f}  kl={kl:.6f}  "
                  f"[finite: rec={rec_ok} kl={kl_ok}]")

            if (not rec_ok) or (not kl_ok) or (loss != loss):
                import numpy as np
                b = {k: np.asarray(v) for k, v in batch.items()}
                print("Non-finite problem; batch statistics:")
                for k in ["images", "joints", "gripper", "target_actions"]:
                    x = b[k]
                    print(f"  {k:15s} shape={x.shape} "
                          f"min={np.nanmin(x):.4g} max={np.nanmax(x):.4g} "
                          f"mean={np.nanmean(x):.4g} any_nan={np.isnan(x).any()} any_inf={np.isinf(x).any()}")
                raise SystemExit("Non-finite detected — training stopped.")

        if step % 1000 == 0:
            mae = eval_step(params, batch_stats, batch)
            if jax.process_index() == 0:
                print(f"eval@{step}: MAE={float(mae[0]):.6f}")

        if step % 500 == 0 and step > 0 and jax.process_index() == 0:
            try:
                host_params      = tree_util.tree_map(lambda x: x[0], params)
                host_batch_stats = tree_util.tree_map(lambda x: x[0], batch_stats)
                checkpoints.save_checkpoint(
                    ckpt_dir=str(CKPT_DIR),
                    target={"params": host_params, "batch_stats": host_batch_stats},
                    step=step, overwrite=True
                )
                print(f"checkpoint saved @ step {step} -> {CKPT_DIR}")
            except Exception as e:
                print(f"[warn] checkpoint save failed at step {step}: {e}")

if __name__ == "__main__":
    main()
