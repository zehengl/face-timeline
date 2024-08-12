from os import getenv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

selfies = Path(getenv("selfies"))
assert selfies.exists()

subfolder = getenv("subfolder")

width = int(getenv("width", "512"))
base_fps = int(getenv("base_fps", "24"))
date_x = float(getenv("date_x", ".75"))
subfolder = getenv("subfolder")

output = Path("output") if not subfolder else Path(f"output/{subfolder}")
output.mkdir(exist_ok=True, parents=True)

faces_yolo = output / "faces-yolo"
faces_yolo.mkdir(exist_ok=True)

faces_cv2 = output / "faces-cv2"
faces_cv2.mkdir(exist_ok=True)

force_all = getenv("force_all") in ["true", "yes"]
