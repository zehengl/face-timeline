# %%
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

output = Path("output")
output.mkdir(exist_ok=True)

faces = output / "faces_cv2"
faces.mkdir(exist_ok=True)

df = pd.read_pickle(output / "df.pkl")


# %%
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

    if len(faces) > 1:
        return None

    for x, y, w, h in faces:
        offset_x = int(w * 0.25)
        offset_y = int(h * 0.25)
        face = img[y - offset_y : y + h + offset_y, x - offset_x : x + w + offset_x]
        cv2.imwrite(str(dest / f"{path.stem}.jpg"), face)


# %%
for path in tqdm(df["file"], desc="Getting Faces by CV2"):
    get_face(path)
