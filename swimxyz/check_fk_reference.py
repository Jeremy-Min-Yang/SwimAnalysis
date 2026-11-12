"""One-time check: smpl_fk.py joints == official smplx joints (needs torch + smplx, kept out of requirements.txt).

    python -m venv /tmp/refvenv && /tmp/refvenv/bin/pip install torch smplx scipy
    /tmp/refvenv/bin/python check_fk_reference.py data/smpl_model/basicmodel_m_lbs_10_207_0_v1.1.0.pkl
Result 2026-11-12: max difference 0.001 mm on freestyle_1, backstroke_7, butterfly_30, breaststroke_60.
"""
import os, pickle, sys, tempfile

import numpy as np
import smplx
import torch

import smpl_fk as fk

raw = fk._Unpickler(open(sys.argv[1], "rb"), encoding="latin1").load()
clean = {k: np.asarray(getattr(raw[k], "x", raw[k])) for k in ("v_template", "shapedirs", "kintree_table", "posedirs", "weights", "f")}
clean["J_regressor"] = raw["J_regressor"]
with tempfile.TemporaryDirectory() as tmp:  # chumpy-free copy for smplx; deleted afterwards (licensed data)
    path = os.path.join(tmp, "SMPL_MALE.pkl")
    pickle.dump(clean, open(path, "wb"))
    ref = smplx.SMPL(model_path=path, num_betas=10)
mine = fk.load_model(sys.argv[1])
t = lambda a: torch.tensor(a, dtype=torch.float32)
worst = 0
for s, i in [("freestyle", 1), ("backstroke", 7), ("butterfly", 30), ("breaststroke", 60)]:
    d = np.load(f"data/smpl/{s}/{s}_{i}_poses.npz")
    P, T = d["poses"], len(d["poses"])
    out = ref(global_orient=t(P[:, :3]), body_pose=t(np.concatenate([P[:, 3:66], np.zeros((T, 6))], 1)),
              betas=t(np.tile(d["betas"][:10], (T, 1))), transl=t(d["trans"]))
    err = np.abs(out.joints[:, :24].detach().numpy() - fk.joints(mine, P, d["betas"], d["trans"])).max()
    worst = max(worst, err)
    print(f"{s}_{i}: max |smplx - smpl_fk| = {err * 1000:.3f} mm")
assert worst < 1e-3, "smpl_fk disagrees with the reference SMPL implementation"
print("ok: smpl_fk matches smplx")
