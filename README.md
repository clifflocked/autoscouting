# Autoscouting
A system for scouting and compiling data from FRC matches and competitions.

## Setup
The main setup is intended to be run on a machine running Lambda stack.
1. Install uv, setup virtual environment, and install packages
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install inference supervision numpy dotenv
```
2. For GPU support, install inference-gpu as well:
```sh
uv pip install inference-gpu
```
3. Run `track.py`
```sh
python3 track.py video.mp4
```

## Getting videos
It is on you to get videos. I recommend using yt-dlp to download some from youtube.
