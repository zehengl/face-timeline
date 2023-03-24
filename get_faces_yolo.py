# %%
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

output = Path("output")
output.mkdir(exist_ok=True)

faces = output / "faces_yolo"
faces.mkdir(exist_ok=True)

df = pd.read_pickle(output / "df.pkl")
model = YOLO("yolov8n.pt")


# %%
for file in tqdm(df["file"].tolist()):
    result = model(file, save=True, classes=[0], verbose=False)

    x, y, w, h = [int(d) for d in result[0].boxes.xywh[0].tolist()]
    r = min(w, h) // 2
    img = cv2.imread(str(file))
    cv2.imwrite(
        str(faces / f"{file.stem}.jpg"),
        img[y - r : y + r, x - r : x + r],
    )
