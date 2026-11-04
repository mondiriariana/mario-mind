"""Per-scan (TR) features for brain encoding, one file per fMRI run (CPU).

For every scan of every run (timing: mjg.encoding.scan_table), from the frames of the play on screen inside
the scan window [f0, f1):
  vjepa       [4, 1664]  V-JEPA 2.1 G rows whose clip ends inside the window (all rows, masks not applied:
                         the screen the person saw), mean over the 8x8 grid then over rows; blocks 11/23/37/47
  vjepa_last  [1664]     same, newest tubelet only (.last.npy)
  buttons     [12]       per button A, B, LEFT, RIGHT, UP, DOWN: share of frames pressed, number of presses
  gamestate   [13]       from RAM (features/ram_full), mean over the window: Mario x / y speed, airborne,
                         big, power state, x on screen, y, absolute x, x gained, dying, enemies on screen,
                         nearest enemy |dx| and dy
  level, t_play (s since the play started), coverage (share of the scan inside the play), stem
Extra band (--extra lowlevel): .lowlevel.npy [T, 3]: luminance, optical flow, audio
envelope from brain_pred/features/lowlevel.py, mean over the window; nuisance band with encode.py --lowlevel.
Scans outside any play: zeros and coverage 0 (the encoding drops them).

  python build_tr_features.py --subjects sub-01      # -> $MJG_ROOT/results/brain_encoding/tr_features/<sub>/<run>.npz
  python build_tr_features.py --subjects sub-01 --extra lowlevel
"""
import argparse, os, re, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from mjg import encoding as E  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
FEAT = f"{ROOT}/features/vjepa21G_c16s8_f8_g8"
RAM = f"{ROOT}/features/ram_full"
LOWLEVEL = f"{ROOT}/features/lowlevel"
OUT = f"{ROOT}/results/brain_encoding/tr_features"
BTN = ["A", "B", "LEFT", "RIGHT", "UP", "DOWN"]


def s8(v):
    v = v.astype(np.int32)
    return np.where(v > 127, v - 256, v)


def gamestate(ram):
    """RAM [n, 2048] -> [n, 13] per frame."""
    mx = ram[:, 0x6D].astype(np.int32) * 256 + ram[:, 0x86]
    scroll = ram[:, 0x71A].astype(np.int32) * 256 + ram[:, 0x71C]
    my = ram[:, 0xCE].astype(np.float32)
    # the 5 enemy slots: drawn flag, x (page * 256 + x in page), y
    drawn = ram[:, 0x0F:0x0F + 5] != 0
    ex = ram[:, 0x6E:0x6E + 5].astype(np.int32) * 256 + ram[:, 0x87:0x87 + 5]
    ey = ram[:, 0xCF:0xCF + 5].astype(np.float32)
    dx = np.where(drawn, np.abs(ex - mx[:, None]), 10_000)
    j = dx.argmin(1)
    near_dx = np.where(drawn.any(1), dx[np.arange(len(j)), j], 256)
    near_dy = np.where(drawn.any(1), ey[np.arange(len(j)), j] - my, 0)
    return np.stack([s8(ram[:, 0x57]) / 40, s8(ram[:, 0x9F]) / 8, ram[:, 0x1D] != 0, ram[:, 0x754] == 0,
                     ram[:, 0x756] / 2, (mx - scroll) / 256, my / 240, mx / 3000, np.r_[0, np.diff(mx)] / 4,
                     np.isin(ram[:, 0x0E], (6, 11)), drawn.sum(1) / 5, np.minimum(near_dx, 256) / 256,
                     near_dy / 240], 1).astype(np.float32)


def play_id(stem):
    """annotation stem (sub-01_ses-001_task-mario_level-w1l1_rep-000) -> feature name (sub-01_ses-001_w1l1_rep-000)"""
    m = re.match(r"(sub-\d+)_(ses-\d+)_task-mario_level-(w\dl\d)_rep-(\d+)", stem)
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_rep-{m.group(4)}" if m else stem


def one_run(job):
    sub, key, st = job
    dst = f"{OUT}/{sub}/{key}.npz"
    if os.path.exists(dst):
        return key, "exists"
    T = len(st)
    vj = np.zeros((T, 4, 1664), np.float32)
    vl = np.zeros((T, 1664), np.float32)
    bt = np.zeros((T, 12), np.float32)
    gs = np.zeros((T, 13), np.float32)
    n_rows = np.zeros(T, np.int32)
    for stem_, g in st[st.stem != ""].groupby("stem"):
        stem = play_id(stem_)
        fpath = f"{FEAT}/{stem}.npy"
        if not os.path.exists(fpath) or not os.path.exists(f"{RAM}/{stem}.npz"):
            continue
        z = np.load(fpath, mmap_mode="r")
        zl = np.load(f"{FEAT}/{stem}.last.npy", mmap_mode="r")
        fi = np.load(f"{FEAT}/{stem}.game.npz")["frame_idx"]
        # written by scripts/data/extract_full_ram.py: button_names is an object array
        rz = np.load(f"{RAM}/{stem}.npz", allow_pickle=True)
        ram, buttons = rz["ram"], rz["buttons"]
        names = []
        for n in rz["button_names"]:
            names.append(str(n))
        gsf = gamestate(ram)
        col = []
        for b in BTN:
            col.append(names.index(b))
        for r in g.itertuples():
            f0, f1 = int(r.f0), int(r.f1)
            if f1 <= f0:
                continue
            rows = np.where((fi >= f0) & (fi < f1))[0]
            if len(rows):
                vj[r.Index_local] = np.asarray(z[rows], np.float32).mean((0, 2))
                vl[r.Index_local] = np.asarray(zl[rows], np.float32).mean((0, 1))
                n_rows[r.Index_local] = len(rows)
            b = buttons[f0:min(f1, len(buttons))][:, col]
            if len(b):
                prev = buttons[f0 - 1][col] if f0 else np.zeros(6, bool)
                on = np.diff(np.vstack([prev[None], b]).astype(int), axis=0) == 1  # presses starting in the window
                bt[r.Index_local] = np.r_[b.mean(0), on.sum(0) / 10]
                gs[r.Index_local] = gsf[f0:min(f1, len(gsf))].mean(0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst + ".tmp.npz", vjepa=vj.astype(np.float16), vjepa_last=vl.astype(np.float16), buttons=bt, gamestate=gs,
             n_rows=n_rows, level=np.array(st.level.astype(str).tolist(), dtype="U8"),
             stem=np.array(st.stem.astype(str).tolist(), dtype="U64"),
             t_play=st.t_play.values.astype(np.float32), coverage=st.coverage.values.astype(np.float32))
    os.replace(dst + ".tmp.npz", dst)
    return key, f"{T} scans, {int((n_rows > 0).sum())} with V-JEPA rows"


def one_run_lowlevel(job):
    """Extra band for the nuisance model: low-level confounds (brain_pred/features/lowlevel.py: luminance, optical flow,
    audio envelope per frame), mean over the scan window's frames. -> <run>.lowlevel.npy [T, 3]"""
    sub, key, st = job
    dst = f"{OUT}/{sub}/{key}.lowlevel.npy"
    if os.path.exists(dst):
        return key, "exists"
    out, missing = np.zeros((len(st), 3), np.float32), 0
    for stem_, g in st[st.stem != ""].groupby("stem"):
        f = f"{LOWLEVEL}/{play_id(stem_)}.npy"
        if not os.path.exists(f):
            missing += 1
            continue
        x = np.load(f)
        for r in g.itertuples():
            if int(r.f1) > int(r.f0):
                out[r.Index_local] = x[int(r.f0):min(int(r.f1), len(x))].mean(0) if int(r.f0) < len(x) else 0
    if missing:  # do not write a partial file: encode.py would treat the missing plays as dark, silent screens
        return key, f"{missing} plays without low-level features, not written"
    np.save(dst, out)
    return key, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=["sub-01"])
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--extra", choices=["lowlevel"], default=None, help="only build this extra band")
    a = ap.parse_args()
    for sub in a.subjects:
        t0 = time.time()
        n_trs = E.run_trs_from_confounds(sub)
        st = E.scan_table(sub, n_trs)
        jobs = []
        for key, g in st.groupby("key"):
            g = g.sort_values("tr").reset_index(drop=True)
            g["Index_local"] = np.arange(len(g))
            jobs.append((sub, key, g))
        n_inside = int((st.coverage > 0.999).sum())
        print(f"{sub}: {len(jobs)} runs, {len(st)} scans, {n_inside} fully inside a play", flush=True)
        fn = one_run_lowlevel if a.extra == "lowlevel" else one_run
        with Pool(a.procs) as pool:
            for key, msg in pool.imap_unordered(fn, jobs):
                print(f"  {key}: {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
