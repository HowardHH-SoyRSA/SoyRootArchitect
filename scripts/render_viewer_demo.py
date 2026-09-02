"""Render a short 60-FPS product demonstration from captured viewer states."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


FPS = 60
DURATION_SECONDS = 10
FRAME_COUNT = FPS * DURATION_SECONDS
WIDTH = 1280
HEIGHT = 720


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _caption(frame: Image.Image, text: str, progress: float, badge: str | None) -> Image.Image:
    output = frame.convert("RGBA")
    overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 612, WIDTH, HEIGHT), fill=(3, 16, 20, 218))
    draw.rectangle((44, 631, 54, 641), fill=(50, 220, 178, 255))
    draw.text((72, 626), text, fill=(239, 248, 246, 255), font=_font(25, bold=True))
    if badge:
        badge_font = _font(18, bold=True)
        bbox = draw.textbbox((0, 0), badge, font=badge_font)
        badge_width = bbox[2] - bbox[0] + 34
        left = WIDTH - badge_width - 44
        draw.rounded_rectangle(
            (left, 625, WIDTH - 44, 653),
            radius=14,
            fill=(13, 59, 224, 210),
            outline=(79, 226, 191, 230),
            width=1,
        )
        draw.text((left + 17, 630), badge, fill=(255, 255, 255, 255), font=badge_font)
    draw.rounded_rectangle((44, 681, WIDTH - 44, 687), radius=3, fill=(38, 61, 66, 255))
    draw.rounded_rectangle(
        (44, 681, 44 + (WIDTH - 88) * max(0.0, min(1.0, progress)), 687),
        radius=3,
        fill=(50, 220, 178, 255),
    )
    return Image.alpha_composite(output, overlay).convert("RGB")


def _timeline_frame(
    images: list[Image.Image],
    captions: list[tuple[float, float, int, str, str | None]],
    timestamp: float,
) -> Image.Image:
    for index, (start, end, image_index, text, badge) in enumerate(captions):
        if start <= timestamp < end or index == len(captions) - 1:
            if index == 0 or timestamp - start >= 0.26:
                base = images[image_index]
            else:
                previous = captions[index - 1]
                blend = max(0.0, min(1.0, (timestamp - start) / 0.26))
                base = Image.blend(images[previous[2]], images[image_index], blend)
            return _caption(base, text, timestamp / DURATION_SECONDS, badge)
    return _caption(images[-1], captions[-1][3], 1.0, captions[-1][4])


def render(input_dir: Path, output_path: Path) -> None:
    names = [
        "11-overview.png",
        "12-orbit-zoom.png",
        "13-root-selected.png",
        "14-assign-ready.png",
        "15-shift-drag-edited.png",
    ]
    images = []
    for name in names:
        image = Image.open(input_dir / name).convert("RGB")
        if image.size != (WIDTH, HEIGHT):
            image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        images.append(image)

    captions = [
        (0.0, 1.7, 0, "SoyRoot Studio · full-resolution 3D root viewer", None),
        (1.7, 3.3, 1, "Orbit and zoom the root system", "VIEW"),
        (3.3, 5.1, 2, "Select a root to highlight and inspect measurements", "INSPECT"),
        (5.1, 6.9, 3, "Activate Assign mode for point ownership edits", "ASSIGN"),
        (6.9, 8.8, 4, "Shift + left-drag paints one continuous region", "SHIFT + LEFT-DRAG"),
        (8.8, 10.0, 4, "The edit is recorded in the undoable history", "480 EDITS"),
    ]

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_index in range(FRAME_COUNT):
            timestamp = frame_index / FPS
            frame = _timeline_frame(images, captions, timestamp)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("The video encoder failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    render(args.input_dir, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
