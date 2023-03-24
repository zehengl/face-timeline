# %%
from pathlib import Path
import cv2
from tqdm import tqdm


output = Path("output")
output.mkdir(exist_ok=True)


# %%
def generate(faces, opt):
    size = (224, 224)
    imgs = []
    desc = f"Generating Video for {opt}"
    for filename in tqdm(sorted(list(faces.glob("*.jpg"))), desc=desc):
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
        str(output / f"face-timeline-{opt}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        size,
    )

    for i in range(len(imgs)):
        out.write(imgs[i])
    out.release()


# %%
for opt in ["cv2", "yolo"]:
    faces = output / f"faces_{opt}"
    faces.mkdir(exist_ok=True)

    generate(faces, opt)
