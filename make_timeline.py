# %%
import logging

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from moviepy import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
from ultralytics import YOLO

from settings import (
    base_fps,
    date_x,
    faces_cv2,
    faces_yolo,
    force_all,
    output,
    selfies,
    width,
)

logging.getLogger("ultralytics").setLevel(logging.ERROR)

model = YOLO("yolo11n.pt")


def get_face(path, dest=faces_cv2):
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


def generate(faces):
    size = (width, width)

    files = list(faces.glob("*.jpg"))
    df = pd.DataFrame({"file": files})
    df["date"] = df["file"].apply(lambda x: pd.to_datetime(x.stem))
    df["year"] = df["date"].dt.year
    df = df.sort_values("date")

    ave_imgs = df.groupby("year").count().reset_index().mean()["file"]

    for year, by_year in df.groupby("year"):
        imgs = []
        desc = f"Generating {year} Video"
        for filename in tqdm(by_year["file"], desc=desc):
            img = cv2.imread(str(filename))
            if img.shape != size:
                img = cv2.resize(img, size)
            img = cv2.putText(
                img,
                filename.stem,
                (int(width * date_x), int(width * 0.9)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            imgs.append(img)

        timeline = output / "face-timeline"
        timeline.mkdir(exist_ok=True)

        out = cv2.VideoWriter(
            str(timeline / f"face-timeline-{year}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(
                int(base_fps * (by_year.shape[0] / ave_imgs)),
                base_fps // 2,
            ),
            size,
        )

        for i in range(len(imgs)):
            out.write(imgs[i])
        out.release()

    print()
    clips = [
        VideoFileClip(str(timeline / f"face-timeline-{y}.mp4"))
        for y in df.year.unique()
    ]

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(str(output / "face-timeline.mp4"), logger=None)


# %%
files = list(selfies.glob("*.jpg"))
df = pd.DataFrame({"file": files})
df["date"] = df["file"].apply(lambda x: pd.to_datetime(x.stem))

# %%
plt.cla()
ax = sns.countplot(df, x=df["date"].dt.year)
ax.set(xlabel="Year")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_figure().savefig(output / "count-per-year.png", dpi=300, bbox_inches="tight")

# %%
plt.cla()
ax = sns.countplot(df, x=df["date"].dt.month)
ax.set(xlabel="Month")
ax.get_figure().savefig(output / "count-per-month.png", dpi=300, bbox_inches="tight")

# %%
plt.cla()
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
ax.get_figure().savefig(output / "count-per-weekday.png", dpi=300, bbox_inches="tight")

# %%
if force_all:
    files = list(selfies.glob("*.jpg"))
else:
    files = [
        f
        for f in selfies.glob("*.jpg")
        if f.name not in [d.name for d in faces_yolo.glob("*.jpg")]
    ]

for file in tqdm(files, desc="Getting Faces by YOLO"):
    result = model(file, classes=[0])

    x, y, w, h = [int(d) for d in result[0].boxes.xywh[0].tolist()]
    r = min(w, h) // 2
    img = cv2.imread(str(file))
    cv2.imwrite(
        str(faces_yolo / f"{file.stem}.jpg"),
        img[y - r : y + r, x - r : x + r],
    )

# %%
if force_all:
    files = list(faces_yolo.glob("*.jpg"))
else:
    files = [
        f
        for f in faces_yolo.glob("*.jpg")
        if f.name not in [d.name for d in faces_cv2.glob("*.jpg")]
    ]

for path in tqdm(files, desc="Getting Faces by CV2"):
    get_face(path)

# %%
generate(faces_cv2)
