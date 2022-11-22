<div align="center">
    <img src="https://cdn1.iconfinder.com/data/icons/ios-11-glyphs/30/face_ID-512.png" alt="logo" height="196">
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

Use `pip install -r requirements-dev.txt` for development.
It will install `pylint` and `black` to enable linting and auto-formatting.
It will also install `jupyterlab` for notebook experience.

## Usage

1.  Create a `.env` file and specify the `selfies` path

        # .env
        selfiles="..."

2.  Run Script

        python make_timeline.py

## Credits

- [Logo][1]

[1]: https://www.iconfinder.com/icons/2639811/face_id_icon
