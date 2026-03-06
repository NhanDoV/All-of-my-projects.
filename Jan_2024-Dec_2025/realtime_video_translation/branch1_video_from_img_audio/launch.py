import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    ImageClip, CompositeVideoClip,
    AudioFileClip, concatenate_videoclips,
    concatenate_audioclips
)
from moviepy.audio.fx import all as afx
from moviepy.video.fx import all as vfx

SCENES = [
    # Scene 1
    {
        "image": "assets/images/img01.jpg",
        "audio": ["assets/audio/aud01.mp3"],
        "text": "Làng Trốn Vò, năm 602"
    },
    {
        "image": "assets/images/img02.jpg",
        "audio": ["assets/audio/aud02.mp3"]
    },

    # Scene 2
    {
        "image": "assets/images/img03.jpg",
        "audio": ["assets/audio/aud03.mp3"],
        "text": "Con chó tên Tô"
    },
    {
        "image": "assets/images/img04.jpg",
        "audio": ["assets/audio/aud04.mp3"]
    },

    # Scene 3
    {
        "image": "assets/images/img05.jpg",
        "audio": ["assets/audio/aud05.mp3"]
    },
    {
        "image": "assets/images/img06.jpg",
        "audio": ["assets/audio/aud06.mp3"],
        "text": "Thần chỉ phạt người"
    },

    # Scene 4
    {
        "image": "assets/images/img07.jpg",
        "audio": ["assets/audio/aud07.mp3"]
    },
    {
        "image": "assets/images/img08.jpg",
        "audio": ["assets/audio/aud08.mp3"]
    },

    # Scene 5
    {
        "image": "assets/images/img09.jpg",
        "audio": ["assets/audio/aud09.mp3"]
    },
    {
        "image": "assets/images/img10.jpg",
        "audio": ["assets/audio/aud10.mp3"]
    },

    # Scene 6
    {
        "image": "assets/images/img11.jpg",
        "audio": ["assets/audio/aud11.mp3"]
    },
    {
        "image": "assets/images/img12.jpg",
        "audio": ["assets/audio/aud12.mp3", "assets/audio/aud13.mp3"],
        "text": "Luật nào cũng có kẽ hở",
        "end": True
    }
]

# =====================
# CONFIG
# =====================
FPS = 24
MIN_SHOT_DURATION = 5.0
TEXT_DELAY = 0.8
TEXT_MIN_DURATION = 1.2
TEXT_MAX_DURATION = 2.0
GAP_DURATION = 0.6


# =====================
# FONT (COLAB SAFE)
# =====================
def load_font(fontsize):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except:
        return ImageFont.load_default()


# =====================
# AUDIO
# =====================
def fit_audio(paths):
    clips = [AudioFileClip(p) for p in paths]
    audio = concatenate_audioclips(clips)
    return audio


# =====================
# TEXT CLIP (NO IMAGEMAGICK)
# =====================
def make_text_clip(text, img_w, img_h, start, duration):
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fontsize = max(28, img_w // 26)
    font = load_font(fontsize)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (img_w - text_w) // 2
    y = int(img_h * 0.82)

    # shadow
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    clip = (
        ImageClip(np.array(img))
        .set_start(start)
        .set_duration(duration)
        .set_fps(FPS)
    )

    # random fade
    fade_in = random.uniform(0.3, 0.6)
    fade_out = random.uniform(0.3, 0.6)

    return (
        clip
        .fx(vfx.fadein, fade_in)
        .fx(vfx.fadeout, fade_out)
    )


# =====================
# BUILD SHOT
# =====================
def build_shot(cfg):
    img = cv2.imread(cfg["image"])
    h, w = img.shape[:2]

    audio = fit_audio(cfg["audio"])
    duration = max(audio.duration, MIN_SHOT_DURATION)

    base = (
        ImageClip(cfg["image"])
        .set_duration(duration)
        .set_fps(FPS)
        .set_audio(audio)
    )

    layers = [base]

    if cfg.get("text"):
        text_duration = random.uniform(TEXT_MIN_DURATION, TEXT_MAX_DURATION)
        text_start = min(TEXT_DELAY, duration - text_duration - 0.2)

        txt_clip = make_text_clip(
            cfg["text"],
            w, h,
            text_start,
            text_duration
        )
        layers.append(txt_clip)

    return CompositeVideoClip(layers)


# =====================
# GAP CLIP (FREEZE FRAME)
# =====================
def make_gap_clip(shot):
    frame = shot.to_ImageClip(t=shot.duration - 0.04)
    return (
        frame
        .set_duration(GAP_DURATION)
        .without_audio()
    )


# =====================
# RENDER
# =====================
def render_video(SCENES):
    clips = []

    for cfg in SCENES:
        shot = build_shot(cfg)
        clips.append(shot)
        clips.append(make_gap_clip(shot))

    final = concatenate_videoclips(clips, method="compose")

    final.write_videofile(
        "output.mp4",
        fps=FPS,
        codec="libx264",
        audio_codec="aac"
    )


# =====================
# RUN
# =====================
render_video(SCENES)