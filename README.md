<div align="center">
    <img src="https://cdn0.iconfinder.com/data/icons/user-pictures/100/unknown_1-512.png" alt="logo" height="128">
</div>

# face-timeline

![coding_style](https://img.shields.io/badge/code%20style-black-000000.svg)

A Python script to make video from selfies

## Environment

- Python 3.9

## Install

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install -r requirements.txt

> Use `pip install -r requirements-dev.txt` for development.

## Usage

1.  Create a `.env` file and specify the `selfies` path

        # .env
        selfies="..."

2.  Run Script

        python prepare.py
        python get_faces_cv2.py
        python get_faces_yolo.py
        python make_timeline.py

## Credits

- [Logo][1] by [Anna Litviniuk][2]

[1]: https://www.iconfinder.com/icons/628286/anonym_avatar_default_head_person_unknown_user_icon
[2]: https://www.iconfinder.com/Naf_Naf
