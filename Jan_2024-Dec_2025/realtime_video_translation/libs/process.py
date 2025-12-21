import subprocess
import sys
import os
from pathlib import Path

def extract_audio_ffmpeg(input_video, output_audio=None, audio_format="m4a"):
    """
        Extract audio from video using FFmpeg.

        Args:
            input_video (str): Path to input video file
            output_audio (str, optional): Output audio filename. Defaults to input basename.
            audio_format (str): Output format (m4a, mp3, wav)
    """
    if not os.path.exists(input_video):
        print(f"Error: {input_video} not found!")
        return False

    input_path = Path(input_video)
    if output_audio is None:
        output_audio = input_path.with_suffix(f".{audio_format}")

    # Basic extraction (copy stream, no re-encoding)
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-vn",  # No video
        "-acodec", "copy",
        "-y",   # Overwrite output
        str(output_audio)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Audio extracted: {output_audio}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False
    
# ==================================== STEP 2 ===============================================
import speech_recognition as sr
from pydub import AudioSegment

# Convert M4A → WAV safely
audio = AudioSegment.from_file("output.m4a")
audio.export("output.wav", format="wav")

r = sr.Recognizer()

# Automatically split long audio into chunks
with sr.AudioFile("output.wav") as source:
    audio_duration = source.DURATION
    print(f"Audio length: {audio_duration:.2f} sec")

texts = []
with sr.AudioFile("output.wav") as source:
    while True:
        audio_chunk = r.record(source, duration=30)  # 30 seconds per chunk
        if len(audio_chunk.frame_data) == 0:
            break
        try:
            part = r.recognize_google(audio_chunk)
            texts.append(part)
        except:
            texts.append("[unrecognized]")

full_text = " ".join(texts)
print("\nFull transcript:")
print(full_text)