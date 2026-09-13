from __future__ import annotations

import json
import math
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Set deterministic seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path("data/synthetic")
IMAGE_DIR = OUTPUT_DIR / "images"
ANNOTATION_FILE = OUTPUT_DIR / "annotations.jsonl"

EQUIPMENT_TYPES = ["electric_motor", "centrifugal_pump", "air_compressor"]
MANUFACTURERS = ["SIEMENS", "ABB", "WEG", "SCHNEIDER", "TOSHIBA", "CROMPTON"]

# Industrial metallic plate colors (BGR)
PLATE_PALETTES = [
    {"bg": (215, 218, 220), "ink": (20, 20, 20), "border": (80, 80, 80)},  # Brushed Aluminum
    {"bg": (200, 205, 210), "ink": (30, 30, 35), "border": (90, 95, 100)},  # Steel
    {"bg": (185, 210, 225), "ink": (25, 25, 25), "border": (120, 140, 150)}, # Anodized Zinc
    {"bg": (220, 220, 215), "ink": (40, 40, 40), "border": (70, 70, 70)},    # Tin plate
]


def random_serial() -> str:
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return (
        f"SN-{''.join(random.choices(chars, k=4))}-"
        f"{''.join(random.choices(chars, k=4))}"
    )


def random_specs(eq_type: str) -> dict:
    voltage = random.choice([220, 230, 240, 380, 400, 415, 460])
    freq = random.choice([50, 60])
    power_kw = random.choice([0.75, 1.5, 2.2, 3.7, 5.5, 7.5, 11.0, 15.0, 22.0])

    # Physically plausible current estimation: I = P / (sqrt(3) * V * pf * eff)
    pf = random.uniform(0.82, 0.88)
    eff = random.uniform(0.85, 0.92)
    current_a = round((power_kw * 1000) / (math.sqrt(3) * voltage * pf * eff), 1)

    rpm_options = [960, 1440, 1475, 2880, 2920] if freq == 50 else [1150, 1750, 3450]
    rpm = random.choice(rpm_options)

    return {
        "equipment_type": eq_type,
        "manufacturer": random.choice(MANUFACTURERS),
        "model": f"{eq_type[:3].upper()}-{random.randint(100, 999)}-{random.choice(['X', 'Y', 'Z'])}",
        "serial_number": random_serial(),
        "voltage": voltage,
        "current": current_a,
        "power": power_kw,
        "frequency": freq,
        "rpm": rpm,
        "phase": "3 PH",
    }


def draw_machinery_background(width: int, height: int) -> np.ndarray:
    """Generates an industrial textured background (cast iron or painted machine shell)."""
    base_color = random.choice([
        (45, 55, 60),    # Industrial Dark Teal/Green
        (50, 50, 55),    # Cast Iron Gray
        (35, 45, 75),    # Industrial Machinery Blue
        (90, 85, 80),    # Faded Protective Enamel
    ])
    bg = np.full((height, width, 3), base_color, dtype=np.uint8)

    # Add surface texture noise
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add gradient shading simulating uneven overhead industrial lighting
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    vignette = 1 - 0.35 * (xx**2 + yy**2)
    bg = np.clip(bg * vignette[:, :, None], 0, 255).astype(np.uint8)
    return bg


def get_scalable_font(size: int) -> ImageFont.FreeTypeFont:
    # Try Windows system font, fallback to standard Linux/Mac paths or default
    font_candidates = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arial.ttf"
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except IOError:
            continue
    return ImageFont.load_default()

def render_plate_text(specs: dict, style_palette: dict) -> Image.Image:
    plate_w, plate_h = 800, 480
    plate = Image.new("RGB", (plate_w, plate_h), style_palette["bg"])
    draw = ImageDraw.Draw(plate)

    draw.rectangle([8, 8, plate_w - 9, plate_h - 9], outline=style_palette["border"], width=3)
    draw.rectangle([14, 14, plate_w - 15, plate_h - 15], outline=style_palette["border"], width=1)

    for bx, by in [(28, 28), (plate_w - 28, 28), (28, plate_h - 28), (plate_w - 28, plate_h - 28)]:
        draw.ellipse([bx - 8, by - 8, bx + 8, by + 8], outline=(90, 90, 90), width=2)

    font_title = get_scalable_font(26)
    font_body = get_scalable_font(20)

    # Header
    draw.text((45, 25), f"MANUFACTURER: {specs['manufacturer']}", fill=style_palette["ink"], font=font_title)
    draw.text((45, 58), f"TYPE: {specs['equipment_type'].upper()} {specs['phase']}", fill=style_palette["ink"], font=font_body)
    draw.line([30, 90, plate_w - 30, 90], fill=style_palette["border"], width=2)

    # Grid rows with readable spacing
    grid_rows = [
        [("MOD", specs["model"]), ("SER NO", specs["serial_number"])],
        [("VOLTS", f"{specs['voltage']} V"), ("AMPS", f"{specs['current']} A")],
        [("RATING", f"{specs['power']} KW"), ("FREQ", f"{specs['frequency']} HZ")],
        [("RPM", f"{specs['rpm']} MIN-1"), ("INS.CL", random.choice(["F", "H", "B"]))],
    ]

    y_pos = 110
    for row in grid_rows:
        x_pos = 45
        for label, val in row:
            draw.rectangle([x_pos - 5, y_pos - 4, x_pos + 330, y_pos + 42], outline=style_palette["border"], width=1)
            draw.text((x_pos + 8, y_pos + 6), f"{label}: {val}", fill=style_palette["ink"], font=font_body)
            x_pos += 360
        y_pos += 60

    return plate


def composite_onto_scene(
    plate_img: Image.Image,
    difficulty: str,
) -> tuple[np.ndarray, list[list[int]]]:
    """Applies realistic perspective skew, glare, and embeds the plate onto a machinery background."""
    canvas_w, canvas_h = 1024, 768
    bg_bgr = draw_machinery_background(canvas_w, canvas_h)

    plate_rgb = np.array(plate_img)
    pw, ph = plate_img.size

    # Source corners (un-warped local plate coordinates)
    src_pts = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])

    # Target corners inside canvas with realistic perspective tilt
    cx, cy = canvas_w // 2, canvas_h // 2
    margin_x, margin_y = pw // 2, ph // 2

    skew_limit = 12 if difficulty == "clean" else (35 if difficulty == "moderate" else 75)

    dst_pts = np.float32([
        [cx - margin_x + random.randint(-skew_limit, skew_limit), cy - margin_y + random.randint(-skew_limit, skew_limit)],
        [cx + margin_x + random.randint(-skew_limit, skew_limit), cy - margin_y + random.randint(-skew_limit, skew_limit)],
        [cx + margin_x + random.randint(-skew_limit, skew_limit), cy + margin_y + random.randint(-skew_limit, skew_limit)],
        [cx - margin_x + random.randint(-skew_limit, skew_limit), cy + margin_y + random.randint(-skew_limit, skew_limit)],
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_plate = cv2.warpPerspective(
        cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2BGR),
        matrix,
        (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Alpha mask for polygon compositing
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)

    # Paste warped plate onto machinery background
    mask_3c = cv2.merge([mask, mask, mask]) // 255
    composite = bg_bgr * (1 - mask_3c) + warped_plate * mask_3c

    # Industrial Degradation Effects
    if difficulty in ["moderate", "severe"]:
        # Specular metallic reflection/glare
        if random.random() < 0.65:
            glare_center = (random.randint(cx - 150, cx + 150), random.randint(cy - 100, cy + 100))
            glare_radius = random.randint(70, 220)
            strength = 75 if difficulty == "moderate" else 135
            yy, xx = np.mgrid[:canvas_h, :canvas_w]
            dist = np.sqrt((xx - glare_center[0]) ** 2 + (yy - glare_center[1]) ** 2)
            glare_mask = np.clip(1.0 - (dist / glare_radius), 0, 1) * strength
            composite = np.clip(composite.astype(np.float32) + glare_mask[:, :, None], 0, 255).astype(np.uint8)

        # Scratches on metallic surface
        num_scratches = 5 if difficulty == "moderate" else 18
        for _ in range(num_scratches):
            p1 = (random.randint(cx - margin_x, cx + margin_x), random.randint(cy - margin_y, cy + margin_y))
            p2 = (p1[0] + random.randint(-60, 60), p1[1] + random.randint(-15, 15))
            cv2.line(composite, p1, p2, (200, 200, 200), thickness=random.choice([1, 2]))

    # Gaussian camera blur / sensor softness
    if difficulty == "moderate":
        composite = cv2.GaussianBlur(composite, (3, 3), 0.5)
    elif difficulty == "severe":
        sigma = random.uniform(0.9, 1.8)
        composite = cv2.GaussianBlur(composite, (5, 5), sigma)

    polygon_corners = dst_pts.astype(int).tolist()
    return composite, polygon_corners


def generate_dataset(num_samples: int = 1000) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} synthetic industrial nameplates...")
    with open(ANNOTATION_FILE, "w", encoding="utf-8") as f_out:
        for idx in range(num_samples):
            eq_type = random.choice(EQUIPMENT_TYPES)
            specs = random_specs(eq_type)
            palette = random.choice(PLATE_PALETTES)

            rand_val = random.random()
            difficulty = "clean" if rand_val < 0.25 else ("moderate" if rand_val < 0.75 else "severe")

            flat_plate = render_plate_text(specs, palette)
            scene_bgr, polygon = composite_onto_scene(flat_plate, difficulty)

            image_name = f"plate_{idx:05d}.jpg"
            image_path = IMAGE_DIR / image_name
            cv2.imwrite(str(image_path), scene_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            record = {
                "image_id": f"plate_{idx:05d}",
                "image_file": image_name,
                "difficulty": difficulty,
                "ground_truth_polygon": polygon,
                "specs": specs,
            }
            f_out.write(json.dumps(record) + "\n")

            if (idx + 1) % 250 == 0 or (idx + 1) == num_samples:
                print(f"  [✓] {idx + 1}/{num_samples} samples generated.")

    print(f"\nCompleted! Annotations saved to: {ANNOTATION_FILE}")


if __name__ == "__main__":
    generate_dataset(500)