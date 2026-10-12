"""Pick a finished KITTI run from the predictions dir and open it in the viser viewer with GT,
or record it to an MP4 (videos/<model>_<seq>.mp4 on scratch).

Usage (in the loger env): python cluster/pick_viser.py [--port 8091]
"""
import argparse
import curses
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_ROOT = Path("/cluster/scratch/fletford/LoGeR/predictions")
GT_ROOT = REPO_ROOT / "data/kitti/dataset/poses"
VIDEO_ROOT = Path("/cluster/scratch/fletford/LoGeR/videos")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8091)
args = parser.parse_args()

# Layout written by cluster/kitti_eval.slurm: <model>/<seq>/results_pi3/*.pt
runs = sorted(PRED_ROOT.glob("*/*/results_pi3/*.pt"))
if not runs:
    sys.exit(f"No predictions found under {PRED_ROOT}")

def menu(title, options):
    """Arrow-key menu with the current line highlighted green; returns the chosen index."""
    def run(stdscr):
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_GREEN)
        idx, top = 0, 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            title_lines = title.split("\n")
            for y, line in enumerate(title_lines):
                stdscr.addnstr(y, 0, line, w - 1)
            rows = max(1, h - len(title_lines) - 2)
            top = min(max(top, idx - rows + 1), idx)
            for y, opt in enumerate(options[top:top + rows]):
                attr = curses.color_pair(1) | curses.A_BOLD if top + y == idx else curses.A_NORMAL
                stdscr.addnstr(len(title_lines) + 1 + y, 0, f" {opt} ".ljust(w - 1), w - 1, attr)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(options)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(options)
            elif key in (curses.KEY_ENTER, 10, 13):
                return idx
            elif key in (ord("q"), 27):
                return None
    choice = curses.wrapper(run)
    if choice is None:
        sys.exit("Cancelled.")
    return choice


rows = []
for pt in runs:
    st = pt.stat()
    date = datetime.fromtimestamp(st.st_mtime).strftime("%m-%d %H:%M")
    rows.append(f"{pt.parts[-4]:<11} {pt.parts[-3]:<4} {st.st_size / 1e9:>6.1f}G  {date}")
header = f"Pick a run (up/down, Enter to open, q to quit)\n\n {'model':<11} {'seq':<4} {'size':>7}  date"
pt = runs[menu(header, rows)]
seq = pt.parts[-3]

# Login nodes cap each user at 12 GB (4 GB soft) and the viewer needs ~3x the file size.
if pt.stat().st_size > 3e9 and socket.gethostname().startswith("eu-login"):
    warning = (f"{pt.stat().st_size / 1e9:.1f} GB is too large for a login node and can kill your SSH session.\n"
               "Run this from an interactive job instead: "
               "srun --cpus-per-task=4 --mem-per-cpu=24G --time=2:00:00 --pty bash")
    if menu(warning, ["Cancel", "Open anyway"]) == 0:
        sys.exit("Cancelled.")

cmd = [sys.executable, str(REPO_ROOT / "demo_viser.py"), "--load", str(pt), "--port", str(args.port)]
gt = GT_ROOT / f"{seq}.txt"
if gt.exists():
    cmd += ["--gt_poses", str(gt)]
else:
    print(f"No GT poses at {gt}; showing estimate only.")

if menu(f"{pt.parts[-4]} seq {seq}: what next?", ["Open in viser", "Save MP4"]) == 1:
    mp4 = VIDEO_ROOT / f"{pt.parts[-4]}_{seq}.mp4"
    cmd += ["--record_mp4", str(mp4)]
    print(f"\nRecording to {mp4}")

print(f"\nOpening {pt.parts[-4]} seq {seq} → forward port {args.port}, then http://localhost:{args.port}\n")
os.chdir(REPO_ROOT)  # demo_viser.py resolves its default ckpt/config paths relative to the cwd
os.execv(cmd[0], cmd)
