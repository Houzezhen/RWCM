"""Extract still frames used by the figures (no image adjustment).

raw/    clean camera frames of one training-distribution episode at the 120-frame-span offsets
video/  frames of the rendered value-prediction videos for the qualitative figure
"""
from pathlib import Path
import json
import subprocess

FF = r"D:/Anaconda3/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
HERE = Path(__file__).resolve().parents[1]
RWCM = HERE.parents[1]
OUT = HERE / "assets"


def grab(src, idx, dst, width=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select=eq(n\\,{idx})" + (f",scale={width}:-1:flags=lanczos" if width else "")
    subprocess.run([FF, "-v", "error", "-y", "-i", str(src), "-vf", vf, "-vsync", "0",
                    "-frames:v", "1", str(dst)], check=True)


def count(src):
    r = subprocess.run([FF, "-i", str(src), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
                       capture_output=True, text=True)
    line = [l for l in r.stderr.splitlines() if "frame=" in l][-1]
    return int(line.split("frame=")[1].split()[0])


manifest = {"raw": [], "video": []}
# Raw memory frames: training-distribution episode 150, current step t=200, 120-frame span offsets
V = RWCM / "data_toiletButton_0814_tail77" / "videos" / "chunk-000"
t = 200
offsets = [0, 17, 34, 51, 69, 86, 103, 120]
for k, off in enumerate(sorted(offsets, reverse=True)):
    for cam in ("primary", "wrist"):
        src = V / f"observation.images.{cam}" / "episode_000150.mp4"
        dst = OUT / "raw" / f"{cam}_{k}.png"
        grab(src, t - off, dst, width=320)
        manifest["raw"].append({"camera": cam, "slot": k, "frame": t - off, "source": str(src.relative_to(RWCM))})

# Rendered value videos
clips = {"ood_st120": RWCM / "assets/videos_wcm_to_spacetime/t120/t120_ood_s3072/episode-000000.mp4",
         "fail_st60": RWCM / "assets/videos_wcm_to_spacetime/t8_pair/fail_spacetime_s3072/episode-000025.mp4"}
for key, src in clips.items():
    n = count(src)
    for k, f in enumerate((0.03, 0.25, 0.5, 0.75, 0.99)):
        idx = round(f * (n - 1))
        grab(src, idx, OUT / "video" / f"{key}_{k}.png")
        manifest["video"].append({"clip": key, "index": idx, "fraction": f, "frames": n,
                                  "source": str(src.relative_to(RWCM))})
(OUT / "frames_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print("ok", len(manifest["raw"]), len(manifest["video"]))
