import os, numpy as np, jax, tensorflow_datasets as tfds

CONTEXT_LEN, CHUNK_LEN = 6, 8

def _norm(u8): return u8.astype(np.float32) / 255.0
def _to32(x):  return x.astype(np.float32)

def iter_episode_steps(ds_name="xgym_sweep_single", split="train", data_dir=None):
    episodes = tfds.load(ds_name, split=split, shuffle_files=True, data_dir=data_dir)
    for ep in episodes:
        yield ep["steps"]  # iterable dataset

def episode_to_arrays(steps_iter, max_steps=None):
    """ steps_iter: TFDS iterable of steps -> dict of arrays [T,...] """
    low, side, wrist = [], [], []
    joints, grip, actions = [], [], []
    n = 0
    for st in steps_iter:
        st = tfds.as_numpy(st)
        # images
        low.append(_norm(st["observation"]["image"]["worm"]))
        side.append(_norm(st["observation"]["image"]["side"]))
        wrist.append(_norm(st["observation"]["image"]["wrist"]))
        # state (proprio)
        joints.append(_to32(st["observation"]["proprio"]["joints"]))
        grip.append(_to32(st["observation"]["proprio"]["gripper"]))
        # action vector (7 joints + 1 gripper)
        aj = st["action"]["joints"]; ag = st["action"]["gripper"]
        actions.append(_to32(np.concatenate([aj, ag], axis=-1)))
        n += 1
        if max_steps is not None and n >= max_steps:
            break
    if n == 0:
        return None
    return {
        "images.low":   np.stack(low,   axis=0),
        "images.side":  np.stack(side,  axis=0),
        "images.wrist": np.stack(wrist, axis=0),
        "state.joints": np.stack(joints,axis=0),
        "state.gripper":np.stack(grip,  axis=0),
        "action":       np.stack(actions,axis=0),  # [T, 8]
    }

def window_samples(seq, context_len=CONTEXT_LEN, chunk_len=CHUNK_LEN):
    T = seq["action"].shape[0]
    for t in range(0, T - (context_len + chunk_len) + 1):
        obs = {
            "images.low":    seq["images.low"][t:t+context_len],
            "images.side":   seq["images.side"][t:t+context_len],
            "images.wrist":  seq["images.wrist"][t:t+context_len],
            "state.joints":  seq["state.joints"][t:t+context_len],
            "state.gripper": seq["state.gripper"][t:t+context_len],
        }
        act = seq["action"][t+context_len:t+context_len+chunk_len]  # [H,8]
        yield obs, act

def stack_cams(obs):
    # [Tctx, H, W, 9]
    return np.concatenate([obs["images.low"], obs["images.side"], obs["images.wrist"]], axis=-1)

def make_batches(ds_name="xgym_sweep_single", split="train", data_dir=None, batch_size=8):
    devices = jax.local_device_count()
    assert batch_size % devices == 0, "batch_size, device sayısına bölünmeli"
    per_dev = batch_size // devices

    def shard(x):
        x = x.reshape(devices, per_dev, *x.shape[1:])
        return jax.device_put_sharded([x[d] for d in range(devices)], jax.local_devices())

    for steps_iter in iter_episode_steps(ds_name, split=split, data_dir=data_dir):
        seq = episode_to_arrays(steps_iter)
        if seq is None: 
            continue
        # control: action dim?
        assert seq["action"].shape[-1] == 8, f"action_dim {seq['action'].shape[-1]} != 8"
        # window samples
        samples = []
        for obs, act in window_samples(seq):
            samples.append((
                {"images": stack_cams(obs), "joints": obs["state.joints"], "gripper": obs["state.gripper"]},
                act
            ))
        # batch and shard
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i+batch_size]
            if len(chunk) < batch_size: break
            imgs   = np.stack([c[0]["images"]  for c in chunk]).astype(np.float32)
            joints = np.stack([c[0]["joints"]  for c in chunk]).astype(np.float32)
            grip   = np.stack([c[0]["gripper"] for c in chunk]).astype(np.float32)
            acts   = np.stack([c[1]            for c in chunk]).astype(np.float32)
            yield {
                "images": shard(imgs),
                "joints": shard(joints),
                "gripper": shard(grip),
                "target_actions": shard(acts),
            }
