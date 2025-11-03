import os, pathlib
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
from jax import tree_util
import numpy as np 

from dataloader_jax import make_batches
from act_model_flax import ACTModel

# ---------------- Env setup ----------------
os.environ.setdefault("TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets"))
os.environ.setdefault("TFDS_WORK_DIR", os.path.expanduser("~/.cache/tfds_work"))
os.makedirs(os.environ["TFDS_WORK_DIR"], exist_ok=True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

# data dir, to be sure
print("TFDS_DATA_DIR =", os.getenv("TFDS_DATA_DIR"))

CKPT_DIR = pathlib.Path("logs/ckpt").expanduser().resolve()
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# --------------- Global ---------------
MODEL = None
optimizer = None

def make_optimizer():
    """AdamW + warmup/cosine, global-norm clip stabilize."""
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
    """Compute scalar beta on the host side (linearly increases from 0 → max_beta)."""
    s = float(step)
    frac = min(1.0, s / float(warmup))
    return max_beta * frac


# ---------------- PMAP ----------------
@partial(jax.pmap, axis_name="dev")
def train_step(params, opt_state, batch, beta_scalar, noise_keys):
    """
    pmap: each device processes its own per-device batch.
    beta_scalar: 0-dimensional scalar (replicated across devices).
    noise_keys: distinct PRNGKey for each device (shape: (2,))
    """
    beta = beta_scalar  # 0-d skaler (replicated)

    # controlled dtype (especially use fp32 for KL computations)
    images  = jnp.asarray(batch["images"], dtype=jnp.float32)
    joints  = jnp.asarray(batch["joints"], dtype=jnp.float32)
    gripper = jnp.asarray(batch["gripper"], dtype=jnp.float32)
    target  = jnp.asarray(batch["target_actions"], dtype=jnp.float32)

    def loss_fn(p):
        pred, kl = MODEL.apply(
            {"params": p},
            images, joints, gripper,
            actions=target, train=True,
            rngs={"noise": noise_keys},  # <-- different RNG for each device
        )

        # Make numerically safe: replace NaN/±∞ with finite values
        pred = jnp.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        kl   = jnp.nan_to_num(kl,   nan=0.0, posinf=1e6, neginf=0.0)

        # rec: MSE (B, chunk, A) -> (B)
        rec = jnp.mean((pred - target) ** 2, axis=(1, 2))
        rec_mean = jnp.mean(rec)
        kl_mean  = jnp.mean(jnp.clip(kl, 0.0, 1e6))

        loss = rec_mean + beta * kl_mean

        # finite flags
        rec_ok = jnp.all(jnp.isfinite(rec))
        kl_ok  = jnp.all(jnp.isfinite(kl))
        return loss, {"rec": rec_mean, "kl": kl_mean, "rec_ok": rec_ok, "kl_ok": kl_ok}

    # loss, aux metrics + grad
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # multidevice mean
    loss    = jax.lax.pmean(loss,    axis_name="dev")
    metrics = jax.lax.pmean(metrics, axis_name="dev")
    grads   = jax.lax.pmean(grads,   axis_name="dev")

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, {"loss": loss, **metrics}

@partial(jax.pmap, axis_name="dev")
def eval_step(params, batch):
    images  = jnp.asarray(batch["images"], dtype=jnp.float32)
    joints  = jnp.asarray(batch["joints"], dtype=jnp.float32)
    gripper = jnp.asarray(batch["gripper"], dtype=jnp.float32)
    target  = jnp.asarray(batch["target_actions"], dtype=jnp.float32)

    pred, _ = MODEL.apply({"params": params},
                          images, joints, gripper,
                          actions=None, train=False)
    mse = jnp.mean((pred - target) ** 2)
    return jax.lax.pmean(mse, axis_name="dev")

# -------------------- Main script --------------------
def main():
    global MODEL, optimizer

    # 1) First batch (init & shape info)
    batch_size = 4   # must be divide evenly by the number of devices
    assert batch_size % jax.local_device_count() == 0, "batch_size, must divide evenly by the number of devices.."
    it = make_batches(batch_size=batch_size)
    first = next(it)

    # First-batch sanity check (one-time)
    _im = np.asarray(first["images"])
    _ac = np.asarray(first["target_actions"])
    print(f"[DATA] images {_im.shape} min={_im.min():.4f} max={_im.max():.4f}")
    print(f"[DATA] actions {_ac.shape} min={_ac.min():.4f} max={_ac.max():.4f}")

    act_dim   = first["target_actions"].shape[-1]
    chunk_len = first["target_actions"].shape[2]

    # Model setup
    # MODEL = ACTModel(action_dim=act_dim, chunk_len=chunk_len)
    # MODEL = ACTModel(action_dim=act_dim, chunk_len=chunk_len, enc_type="vit")
    MODEL = ACTModel(action_dim=act_dim, chunk_len=chunk_len, enc_type="vit_b16", freeze_backbone=True)

    # 2) Initialize OUTSIDE PMAP with a single-device slice
    single = {k: val[0] for k, val in first.items()}  # [per_dev, ...]
    variables = MODEL.init({"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(0)},
                           single["images"], single["joints"], single["gripper"],
                           actions=single["target_actions"])
    params = variables["params"]

    # (optional) Load from checkpoint (absolute path required)
    restored = checkpoints.restore_checkpoint(ckpt_dir=str(CKPT_DIR), target={"params": params})
    params = restored.get("params", params)

    # 3) Prepare optimizer & replicate
    optimizer_local = make_optimizer()
    opt_state = optimizer_local.init(params)
    params    = jax.device_put_replicated(params, jax.local_devices())
    opt_state = jax.device_put_replicated(opt_state, jax.local_devices())
    globals()["optimizer"] = optimizer_local  # for pmap closure

    # RNG root key
    rng = jax.random.PRNGKey(0)

    # 4) Training
    total_steps = 2_000
    log_every   = 5

    for step in range(total_steps):
        beta = beta_anneal(step, warmup=1_000, max_beta=1e-3)
        beta_vec = jnp.full((jax.local_device_count(),), beta, dtype=jnp.float32)

        # her adımda yeni RNG: per-device anahtarlar
        rng, step_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, jax.local_device_count())
        step_keys = jax.device_put_sharded(list(step_keys), jax.local_devices())

        batch = first if step == 0 else next(it)

        params, opt_state, metrics = train_step(params, opt_state, batch, beta_vec, step_keys)

        if (step % log_every == 0) and (jax.process_index() == 0):
            m = jax.device_get(metrics)   # replicated -> host
            loss = float(m["loss"][0])    # [n_devices,]; [0] 
            rec  = float(m["rec"][0])
            kl   = float(m["kl"][0])
            rec_ok = bool(m["rec_ok"][0])
            kl_ok  = bool(m["kl_ok"][0])
            # Print KL in scientific notation so small values are visible
            # print(f"step {step:5d}  beta={float(beta):.6g}  loss={loss:.6f}  rec={rec:.6f}  kl={kl:.6f}  "
            print(f"step {step:5d}  beta={float(beta):.9g}  loss={loss:.9f}  rec={rec:.9f}  kl={kl:.9e}  "
                  f"[finite: rec={rec_ok} kl={kl_ok}]")

            # Non-finite values appear, make detection easier
            if (not rec_ok) or (not kl_ok) or (loss != loss):
                b = {k: np.asarray(v) for k, v in batch.items()}
                print("⚠️ Non-finite; batch statistics:")
                for k in ["images", "joints", "gripper", "target_actions"]:
                    x = b[k]
                    print(f"  {k:15s} shape={x.shape} "
                          f"min={np.nanmin(x):.4g} max={np.nanmax(x):.4g} "
                          f"mean={np.nanmean(x):.4g} any_nan={np.isnan(x).any()} any_inf={np.isinf(x).any()}")
                raise SystemExit("Non-finite detected — training stopped.")

        # Eval (fast; on train batch) — empty/NaN must be 
        if step % 1000 == 0:
            try:
                mse = eval_step(params, batch)
                if jax.process_index() == 0:
                    print(f"eval@{step}: MSE={float(mse[0]):.6f}")
            except Exception as _e:
                if jax.process_index() == 0:
                    print(f"eval@{step}: SKIP ({_e})")

        # Checkpoint
        if step % 500 == 0 and step > 0 and jax.process_index() == 0:
            try:
                host_params = tree_util.tree_map(lambda x: x[0], params)  # replica starts from 0
                checkpoints.save_checkpoint(ckpt_dir=str(CKPT_DIR),
                                            target={"params": host_params},
                                            step=step, overwrite=True)
                print(f"checkpoint saved @ step {step} -> {CKPT_DIR}")
            except Exception as e:
                print(f"[warn] checkpoint save failed at step {step}: {e}")

if __name__ == "__main__":
    main()
