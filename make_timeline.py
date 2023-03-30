# %%
from pathlib import Path

import cv2
import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

output = Path("output")
output.mkdir(exist_ok=True)


# %%
def generate(faces, opt):
    size = (224, 224)

    files = list(faces.glob("*.jpg"))
    df = pd.DataFrame({"file": files})
    df["date"] = df["file"].apply(lambda x: pd.to_datetime(x.stem))
    df["year"] = df["date"].dt.year
    df = df.sort_values("date")

    ave_imgs = df.groupby("year").count().reset_index().mean()["file"]
    base_fps = 24

    for year, by_year in df.groupby("year"):
        imgs = []
        desc = f"Generating {year} Video for {opt}"
        for filename in tqdm(by_year["file"], desc=desc):
            img = cv2.imread(str(filename))
            if img.shape != size:
                img = cv2.resize(img, size)
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
            str(output / f"face-timeline-{opt}-{year}.mp4"),
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
        VideoFileClip(f"output/face-timeline-{opt}-{y}.mp4") for y in df.year.unique()
    ]

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(f"output/face-timeline-{opt}.mp4", logger=None)


# %%
for opt in ["cv2", "yolo"]:
    faces = output / f"faces_{opt}"
    faces.mkdir(exist_ok=True)

    generate(faces, opt)
