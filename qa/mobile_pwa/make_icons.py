"""Generate the home-screen icons of the installable web app from assets/logo.png.

    python qa/mobile_pwa/make_icons.py

Writes assets/icons/: icon-192.png and icon-512.png (purpose "any"), icon-maskable-192.png and icon-maskable-512.png (the
logo inside the central safe zone, so a round or squircle mask never cuts it) and apple-touch-icon.png (180 px, opaque).
Run it again whenever the logo changes. Needs Pillow (already in requirements.txt).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LOGO = ROOT / "assets" / "logo.png"
OUT = ROOT / "assets" / "icons"
WHITE = (255, 255, 255, 255)


def icon(size: int, logo_share: float) -> Image.Image:
    """A white square of ``size`` px with the logo centred, its longer side ``logo_share`` of the square."""
    logo = Image.open(LOGO).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    scale = size * logo_share / max(logo.size)
    logo = logo.resize((max(1, round(logo.width * scale)), max(1, round(logo.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), WHITE)
    canvas.alpha_composite(logo, ((size - logo.width) // 2, (size - logo.height) // 2))
    return canvas.convert("RGB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, size, share in (("icon-192.png", 192, 0.84), ("icon-512.png", 512, 0.84), ("icon-maskable-192.png", 192, 0.62),
                              ("icon-maskable-512.png", 512, 0.62), ("apple-touch-icon.png", 180, 0.80)):
        icon(size, share).save(OUT / name, optimize=True)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
