# %%
from pathlib import Path
from os import getenv

import cv2
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

output = Path("output")
output.mkdir(exist_ok=True)

faces_folder = Path("output/faces")
faces_folder.mkdir(exist_ok=True)

selfies = Path(getenv("selfies"))
assert selfies.exists()

# %%
files = list(selfies.glob("*.jpg"))
df = pd.DataFrame({"file": files})
df["date"] = df["file"].apply(lambda x: pd.to_datetime(x.stem))

# %%
ax = sns.countplot(df, x=df["date"].dt.year)
ax.set(xlabel="Year")
ax.get_figure().savefig("output/count-per-year.png", dpi=300, bbox_inches="tight")

# %%
ax = sns.countplot(df, x=df["date"].dt.month)
ax.set(xlabel="Month")
ax.get_figure().savefig("output/count-per-month.png", dpi=300, bbox_inches="tight")

# %%
ax = sns.countplot(
    df,
    x=df["date"].dt.day_name(),
    order=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)
ax.set(xlabel="Weekday")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.get_figure().savefig("output/count-per-weekday.png", dpi=300, bbox_inches="tight")

# %%
size = (224, 224)


def get_face(path, dest=faces_folder):
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

    for (x, y, w, h) in faces:
        offset_x = int(w * 0.25)
        offset_y = int(h * 0.25)
        face = img[y - offset_y : y + h + offset_y, x - offset_x : x + w + offset_x]
        face = cv2.resize(face, size)
        cv2.imwrite(str(dest / f"{path.stem}.jpg"), face)


# %%
for path in tqdm(df["file"], desc="Getting Faces"):
    get_face(path)

# %%
imgs = []
for filename in tqdm(sorted(list(faces_folder.glob("*.jpg"))), desc="Generating Video"):
    img = cv2.imread(str(filename))
    img = cv2.putText(
        img,
        filename.stem,
        (130, 220),
        cv2.FONT_HERSHEY_COMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    imgs.append(img)

out = cv2.VideoWriter(
    str(output / "face-timeline.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, size
)

for i in range(len(imgs)):
    out.write(imgs[i])
out.release()

# %%
