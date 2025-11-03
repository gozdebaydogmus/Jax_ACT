import os, numpy as np, tensorflow_datasets as tfds

DS = "xgym_sweep_single"
data_dir = os.getenv("TFDS_DATA_DIR")

# Episode-level dataset 
episodes = tfds.load(DS, split="train", shuffle_files=False, data_dir=data_dir)

# First episode
ep = next(iter(episodes))
steps = ep["steps"]  # 

# First 16, just example
low_list, side_list, wrist_list = [], [], []
joints_list, grip_list, act_list = [], [], []

for i, st in enumerate(steps.take(16)):
    st = tfds.as_numpy(st)

    # observation.image
    low   = st["observation"]["image"]["worm"]   # 64x64x3
    side  = st["observation"]["image"]["side"]
    wrist = st["observation"]["image"]["wrist"]

    # Proprio
    joints = st["observation"]["proprio"]["joints"]   # (7,)
    grip   = st["observation"]["proprio"]["gripper"]  # (1,)

    # Action
    aj = st["action"]["joints"]    # (7,)
    ag = st["action"]["gripper"]   # (1,)
    a  = np.concatenate([aj, ag], axis=-1)  # (8,)

    low_list.append(low); side_list.append(side); wrist_list.append(wrist)
    joints_list.append(joints); grip_list.append(grip); act_list.append(a)
    # just few steps
    if i == 0:
        print("low :", low.shape, low.dtype)
        print("side:", side.shape, side.dtype)
        print("wrist:", wrist.shape, wrist.dtype)
        print("state.joints:", joints.shape, joints.dtype, "gripper:", grip.shape, grip.dtype)
        print("action vec (7+j1):", a.shape, a.dtype)

print(f"\nCollected {len(low_list)} steps from first episode.")
