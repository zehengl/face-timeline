# %%
import logging
from os import getenv
from pathlib import Path

import cv2
from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

load_dotenv(override=True)

subfolder = getenv("subfolder")

output = Path("output") if not subfolder else Path(f"output/{subfolder}")
output.mkdir(exist_ok=True, parents=True)

selfies = Path(getenv("selfies"))

faces_yolo = output / "faces-yolo"
faces_yolo.mkdir(exist_ok=True)

model = YOLO("yolov8n.pt")

# %%
for file in tqdm(list(selfies.glob("*.jpg")), desc="Getting Faces by YOLO"):
    result = model(file, classes=[0])

    x, y, w, h = [int(d) for d in result[0].boxes.xywh[0].tolist()]
    r = min(w, h) // 2
    img = cv2.imread(str(file))
    cv2.imwrite(
        str(faces_yolo / f"{file.stem}.jpg"),
        img[y - r : y + r, x - r : x + r],
    )


# %%
faces = output / "faces-cv2"
faces.mkdir(exist_ok=True)


def get_face(path, dest=faces):
    img = cv2.imread(str(path))
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    height, width, _ = img.shape
    faces = face_cascade.detectMultiScale(gray)

    faces = [
        (x, y, w, h) for x, y, w, h in faces if w > 0.2 * width and h > 0.2 * height
    ]

    if not faces:
        return None

    face = max(faces, key=lambda t: t[-1] * t[-2])

    x, y, w, h = face
    offset_x = min(int(w * 0.25), (width - x - w) // 2, x // 2)
    offset_y = min(int(h * 0.25), (height - y - h) // 2, y // 2)
    offset = min(offset_x, offset_y)

    face = img[y - offset : y + h + offset, x - offset : x + w + offset]
    cv2.imwrite(str(dest / f"{path.stem}.jpg"), face)


# %%
for path in tqdm(list(faces_yolo.glob("*.jpg")), desc="Getting Faces by CV2"):
    get_face(path)
