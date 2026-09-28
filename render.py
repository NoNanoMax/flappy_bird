"""PIL-based renderer for Flappy Bird — produces clean frames for video."""

from PIL import Image, ImageDraw, ImageFont


# ── colors ───────────────────────────────────────────────────────────
SKY_TOP = (135, 206, 235)
SKY_BOT = (220, 240, 255)
PIPE_GREEN = (60, 170, 60)
PIPE_BORDER = (40, 120, 40)
PIPE_LIP = (80, 200, 80)
BIRD_YELLOW = (255, 210, 50)
BIRD_WING = (255, 180, 30)
BIRD_EYE = (255, 255, 255)
BIRD_PUPIL = (30, 30, 30)
BIRD_BEAK = (240, 130, 30)
GROUND_DIRT = (180, 140, 80)
GROUND_GRASS = (100, 180, 60)
GROUND_GRASS_DARK = (70, 150, 40)
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)


def render_frame(bird_y: float, pipes: list, score: int,
                 width: int = 288, height: int = 512,
                 ground_h: int = 80, pipe_w: int = 52, pipe_gap: int = 150,
                 bird_x: int = 50, bird_size: int = 16) -> Image.Image:
    """Render one frame. pipes = [(x, gap_center, scored), ...]"""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # ── sky gradient ──
    for y in range(height - ground_h):
        t = y / (height - ground_h)
        r = int(SKY_TOP[0] + (SKY_BOT[0] - SKY_TOP[0]) * t)
        g = int(SKY_TOP[1] + (SKY_BOT[1] - SKY_TOP[1]) * t)
        b = int(SKY_TOP[2] + (SKY_BOT[2] - SKY_TOP[2]) * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # ── pipes ──
    for px, gc, _ in pipes:
        if px > width + pipe_w or px < -pipe_w:
            continue
        gap_top = gc - pipe_gap // 2
        gap_bot = gc + pipe_gap // 2
        ground_y = height - ground_h

        # upper pipe
        draw.rectangle([px, 0, px + pipe_w, gap_top], fill=PIPE_GREEN, outline=PIPE_BORDER)
        # upper pipe lip
        draw.rectangle([px - 4, gap_top - 20, px + pipe_w + 4, gap_top],
                        fill=PIPE_LIP, outline=PIPE_BORDER)

        # lower pipe
        draw.rectangle([px, gap_bot, px + pipe_w, ground_y], fill=PIPE_GREEN, outline=PIPE_BORDER)
        # lower pipe lip
        draw.rectangle([px - 4, gap_bot, px + pipe_w + 4, gap_bot + 20],
                        fill=PIPE_LIP, outline=PIPE_BORDER)

    # ── ground ──
    ground_y = height - ground_h
    draw.rectangle([0, ground_y, width, height], fill=GROUND_DIRT)
    # grass strip
    draw.rectangle([0, ground_y, width, ground_y + 12], fill=GROUND_GRASS)
    # grass texture (simple zigzag)
    for x in range(0, width, 8):
        draw.polygon([(x, ground_y), (x + 4, ground_y + 6), (x + 8, ground_y)],
                     fill=GROUND_GRASS_DARK)

    # ── bird ──
    bx, by = bird_x, int(bird_y)
    # body
    draw.ellipse([bx - bird_size, by - bird_size, bx + bird_size, by + bird_size],
                 fill=BIRD_YELLOW, outline=(200, 170, 30))
    # wing
    draw.ellipse([bx - 8, by - 4, bx + 4, by + 8], fill=BIRD_WING)
    # eye
    draw.ellipse([bx + 4, by - 8, bx + 14, by + 2], fill=BIRD_EYE, outline=(30, 30, 30))
    draw.ellipse([bx + 7, by - 5, bx + 12, by + 0], fill=BIRD_PUPIL)
    # beak
    draw.polygon([(bx + bird_size - 2, by - 2), (bx + bird_size + 8, by + 2),
                  (bx + bird_size - 2, by + 6)], fill=BIRD_BEAK)

    # ── score ──
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text = str(score)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (width - tw) // 2
    ty = 20
    # shadow
    draw.text((tx + 2, ty + 2), text, font=font, fill=TEXT_SHADOW)
    # main
    draw.text((tx, ty), text, font=font, fill=TEXT_COLOR)

    return img
