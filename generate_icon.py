"""Generate assets/app.ico from the tray icon function in app.py."""

import os
from app import create_tray_icon


def main():
    os.makedirs("assets", exist_ok=True)
    sizes = [16, 32, 48, 64, 128, 256]
    images = [create_tray_icon(size=s, speaking=False) for s in sizes]
    images[0].save(
        "assets/app.ico",
        format="ICO",
        append_images=images[1:],
        sizes=[(s, s) for s in sizes],
    )
    print("Created assets/app.ico")


if __name__ == "__main__":
    main()
