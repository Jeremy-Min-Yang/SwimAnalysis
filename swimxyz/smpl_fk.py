"""SMPL forward kinematics in numpy: SwimXYZ motion (.npz) -> 3D joint positions.

Only the 24 skeleton joints are needed, so no mesh skinning and no torch/chumpy.
"""
import pickle

import numpy as np

# SMPL joint names (index = SMPL joint id)
JOINTS = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3",
          "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow",
          "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand"]


class _Chumpy:
    """Stands in for chumpy arrays inside the official SMPL .pkl; keeps only the stored array."""
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): self.x = state["x"]


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _Chumpy if module.startswith("chumpy") else super().find_class(module, name)


def load_model(path):
    with open(path, "rb") as f:
        d = _Unpickler(f, encoding="latin1").load()
    shapedirs = d["shapedirs"]
    return dict(J_regressor=d["J_regressor"].toarray(), v_template=np.asarray(d["v_template"]),
                shapedirs=np.asarray(getattr(shapedirs, "x", shapedirs)),
                parents=d["kintree_table"][0].astype(int))


def rodrigues(r):
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    theta = np.linalg.norm(r, axis=-1, keepdims=True)
    k = r / np.maximum(theta, 1e-8)
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return np.eye(3) + s * K + (1 - c) * K @ K


def rest_joints(model, betas):
    v = model["v_template"] + model["shapedirs"][..., :10] @ betas[:10]  # shortcut: SwimXYZ has 16 SMPL-H betas, SMPL uses the first 10
    return model["J_regressor"] @ v


def joints(model, poses, betas, trans):
    """poses (T, >=66) axis-angle, SMPL-H body order; returns (T, 24, 3) world joints in metres."""
    T = len(poses)
    J = rest_joints(model, betas)
    pose = np.zeros((T, 24, 3))
    pose[:, :22] = poses[:, :66].reshape(T, 22, 3)  # SMPL-H body = SMPL joints 0-21; SMPL hand joints 22-23 stay at rest
    R = rodrigues(pose)
    par = model["parents"]
    GR, Gt = [R[:, 0]], [np.broadcast_to(J[0], (T, 3))]
    for i in range(1, 24):
        p = par[i]
        GR.append(GR[p] @ R[:, i])
        Gt.append(Gt[p] + GR[p] @ (J[i] - J[p]))
    return np.stack(Gt, 1) + trans[:, None]


if __name__ == "__main__":  # self-check: python smpl_fk.py <model.pkl> <motion.npz>
    import sys
    m = load_model(sys.argv[1])
    d = np.load(sys.argv[2])
    J0 = rest_joints(m, d["betas"])
    rest = joints(m, np.zeros((1, 66)), d["betas"], np.zeros((1, 3)))[0]
    assert np.allclose(rest, J0), "zero pose must give the rest skeleton"
    X = joints(m, d["poses"], d["betas"], d["trans"])
    bone = np.linalg.norm(X[:, 1:] - X[:, m["parents"][1:]], axis=-1)
    assert bone.std(0).max() < 1e-9, "bone lengths must not change over time"
    assert np.allclose(rodrigues(np.array([0, 0, np.pi / 2])) @ [1, 0, 0], [0, 1, 0]), "rotation direction"
    print("ok", X.shape, "height (m) ~", round(float(np.ptp(J0[:, 1])), 2),
          "| r_wrist-l_wrist range (m):", np.ptp(np.linalg.norm(X[:, 21] - X[:, 20], axis=1)).round(2))
