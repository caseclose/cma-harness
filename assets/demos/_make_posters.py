"""Generate fancy demo posters: pick the most colorful frame per video and stylize."""
import os
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

FFMPEG = "/root/.local/lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
FFPROBE_ARGS = [FFMPEG, "-hide_banner", "-i"]
FONT_BOLD = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/dejavu/DejaVuSans.ttf"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_W = 1920

def probe_duration(path: str) -> float:
    try:
        out = subprocess.run(FFPROBE_ARGS + [path], capture_output=True, text=True).stderr
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Duration:"):
                t = line.split(",")[0].replace("Duration:", "").strip()
                h, m, s = t.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    return 30.0

def grab_frame(video_path: str, at: float, width: int = 0) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        args = [FFMPEG, "-y", "-ss", f"{at:.2f}", "-i", video_path,
                "-frames:v", "1"]
        if width:
            args += ["-vf", f"scale={width}:-2"]
        args.append(tmp)
        subprocess.run(args, check=True, capture_output=True)
        return Image.open(tmp).convert("RGB")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def frame_saturation(img: Image.Image) -> float:
    """Mean saturation in HSV (0-255). Higher = more colorful."""
    hsv = img.convert("HSV")
    return ImageStat.Stat(hsv).mean[1]

def pick_best_time(video_path: str, dur: float) -> float:
    """Sample frames unevenly (denser in the last third), return the most colorful one."""
    # First two thirds: coarse sampling. Last third: dense sampling (fanciest frames
    # tend to be near the end of the interaction).
    n_early, n_late = 10, 30
    early = [dur * (0.05 + (2/3 - 0.05) * k / max(1, n_early - 1)) for k in range(n_early)]
    late = [dur * (2/3 + (0.97 - 2/3) * k / max(1, n_late - 1)) for k in range(n_late)]
    times = early + late
    best = (0.0, -1.0)
    for t in times:
        try:
            f = grab_frame(video_path, t, width=320)
        except Exception:
            continue
        s = frame_saturation(f)
        if s > best[1]:
            best = (t, s)
    return best[0] if best[1] >= 0 else dur * 0.5

def compose(img: Image.Image, idx: int, subtitle: str) -> Image.Image:
    # resize keeping aspect
    w, h = img.size
    new_w = OUT_W
    new_h = int(h * new_w / w)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # subtle darken + slight vignette base
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # bottom gradient (dark)
    grad_h = int(new_h * 0.55)
    for i in range(grad_h):
        alpha = int(210 * (i / grad_h) ** 1.6)
        draw.rectangle(
            [0, new_h - grad_h + i, new_w, new_h - grad_h + i + 1],
            fill=(6, 10, 22, alpha),
        )

    # accent bar top-left
    draw.rectangle([0, 0, new_w, 6], fill=(59, 130, 246, 220))

    # text
    title_font = ImageFont.truetype(FONT_BOLD, int(new_h * 0.10))
    sub_font = ImageFont.truetype(FONT_REG, int(new_h * 0.042))
    tag_font = ImageFont.truetype(FONT_BOLD, int(new_h * 0.035))

    title = TITLES[idx - 1]
    eyebrow = f"DEMO {idx}"
    padding_x = int(new_w * 0.05)
    baseline_y = new_h - int(new_h * 0.09)

    # eyebrow
    eb_font = ImageFont.truetype(FONT_BOLD, int(new_h * 0.038))
    eb_bbox = draw.textbbox((0, 0), eyebrow, font=eb_font)
    eb_h = eb_bbox[3] - eb_bbox[1]

    # title
    tb = draw.textbbox((0, 0), title, font=title_font)
    th = tb[3] - tb[1]

    # position: eyebrow above title
    title_y = baseline_y - th
    eyebrow_y = title_y - eb_h - int(new_h * 0.02)

    draw.text((padding_x, eyebrow_y), eyebrow, font=eb_font,
              fill=(147, 197, 253, 240))
    draw.text((padding_x, title_y), title, font=title_font,
              fill=(255, 255, 255, 255))

    # subtitle
    draw.text((padding_x, baseline_y + int(new_h * 0.008)), subtitle,
              font=sub_font, fill=(203, 213, 225, 235))

    # top-right tag pill
    tag_text = "CMA-Harness"
    tb2 = draw.textbbox((0, 0), tag_text, font=tag_font)
    tw = tb2[2] - tb2[0]
    th2 = tb2[3] - tb2[1]
    pad_x = int(th2 * 0.9)
    pad_y = int(th2 * 0.55)
    box_w = tw + pad_x * 2
    box_h = th2 + pad_y * 2
    x1 = new_w - int(new_w * 0.035) - box_w
    y1 = int(new_h * 0.045)
    draw.rounded_rectangle(
        [x1, y1, x1 + box_w, y1 + box_h],
        radius=int(box_h * 0.5),
        fill=(255, 255, 255, 235),
    )
    draw.text(
        (x1 + pad_x, y1 + pad_y - int(th2 * 0.15)),
        tag_text, font=tag_font, fill=(15, 23, 42, 255),
    )

    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return out


TITLES = [
    "Bedroom Refresh Planner",
    "Bedroom Makeover Poster",
    "Weeknight Recipe Card",
    "Pet Birthday Composer",
    "Pet Birthday Poster",
    "Second-Hand Listing Studio",
    "AI Stock Watchlist Brief",
    "Brand Logo Fusion",
]

SUBTITLES = [
    "Budget planning, reference images, and shopping list",
    "Room understanding and social poster generation",
    "Meal ideas and stylized recipe card",
    "Pet photo understanding and holiday composition",
    "Cross-turn visual recall for party invitation",
    "Item inspection and marketplace listing image",
    "Real-time search plus financial brief poster",
    "Web search, logo fusion, and brand release image",
]

OVERRIDE_TIMES = {
    2: 246.6,
    4: 129.5,
    7: 145.0,
    8: 340.5,
}

def main():
    for i in range(1, 9):
        video = os.path.join(HERE, f"demo-{i}.mp4")
        if not os.path.exists(video):
            print(f"skip missing {video}")
            continue
        dur = probe_duration(video)
        if i in OVERRIDE_TIMES:
            at = OVERRIDE_TIMES[i]
            src_tag = "override"
        else:
            at = pick_best_time(video, dur)
            src_tag = "auto"
        at = max(1.0, min(at, dur - 1.0))
        print(f"demo-{i}.mp4 duration={dur:.1f}s frame@{at:.1f}s ({src_tag})")
        frame = grab_frame(video, at)
        poster = compose(frame, i, SUBTITLES[i - 1])
        out_path = os.path.join(HERE, f"demo-{i}.jpg")
        poster.save(out_path, "JPEG", quality=88, optimize=True)
        print(f"  -> {out_path} ({os.path.getsize(out_path)//1024} KB)")


if __name__ == "__main__":
    main()
