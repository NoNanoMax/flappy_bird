import subprocess
import sys
from pathlib import Path


def compile_video(frames_dir: str, output: str, fps: int = 150,
                  width: int = 432, height: int = 768):
    """Compile a directory of PNG frames into MP4. Higher fps = faster video."""
    if not Path(frames_dir).exists():
        print(f"  SKIP: {frames_dir} not found")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frames_dir}/%05d.png",
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output,
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"  → {output}")
    return True


def main():
    frames_base = Path("frames")
    output_base = Path("video")
    output_base.mkdir(exist_ok=True)

    if not frames_base.exists():
        print("No frames/ directory found. Run record.py first.")
        sys.exit(1)

    # Compile each episode
    ep_dirs = sorted(frames_base.iterdir())
    if not ep_dirs:
        print("No episode directories found.")
        sys.exit(1)

    for ep_dir in ep_dirs:
        if ep_dir.is_dir():
            out = str(output_base / f"{ep_dir.name}.mp4")
            compile_video(str(ep_dir), out)

    # Create a combined video (all episodes in sequence)
    print("\nCreating combined video...")
    concat_list = output_base / "concat.txt"
    with open(concat_list, "w") as f:
        for ep_dir in sorted(frames_base.iterdir()):
            if ep_dir.is_dir():
                mp4 = output_base / f"{ep_dir.name}.mp4"
                if mp4.exists():
                    f.write(f"file '{mp4.name}'\n")

    combined_out = str(output_base / "combined.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        combined_out,
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"  → {combined_out}")


if __name__ == "__main__":
    main()
