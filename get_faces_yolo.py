# %%
import logging
from os import getenv
from pathlib import Path

import cv2
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

load_dotenv()

subfolder = getenv("subfolder")

output = Path("output") if not subfolder else Path(f"output/{subfolder}")
output.mkdir(exist_ok=True, parents=True)

faces = output / "faces-yolo"
faces.mkdir(exist_ok=True)

df = pd.read_pickle(output / "df.pkl")

model = YOLO("yolov8n.pt")

# %%
for file in tqdm(df["file"], desc="Getting Faces by YOLO"):
    result = model(file, classes=[0])

    x, y, w, h = [int(d) for d in result[0].boxes.xywh[0].tolist()]
    r = min(w, h) // 2
    img = cv2.imread(str(file))
    cv2.imwrite(
        str(faces / f"{file.stem}.jpg"),
        img[y - r : y + r, x - r : x + r],
    )
