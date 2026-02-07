import streamlit as st
import tempfile
import os
import subprocess
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

st.title("🎬 HinSync - English → Hindi Subtitle Generator")

# -------------------------------
# Load Models (Cached)
# -------------------------------

@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu")

@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

whisper_model = load_whisper()
tokenizer, translator_model = load_translator()

# -------------------------------
# Translate Function
# -------------------------------

def translate_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = translator_model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# -------------------------------
# Create SRT
# -------------------------------

def format_time(seconds):
    millis = int((seconds % 1) * 1000)
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

def create_srt(segments, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_time(seg.start)
            end = format_time(seg.end)
            hindi_text = translate_text(seg.text)

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{hindi_text}\n\n")

# -------------------------------
# Upload Video
# -------------------------------

video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv"])

if video_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path)

    if st.button("Generate Subtitles"):

        st.info("Transcribing audio...")

        segments, _ = whisper_model.transcribe(video_path)

        segments = list(segments)

        srt_path = video_path.replace(".mp4", ".srt")

        st.info("Translating and creating subtitles...")
        create_srt(segments, srt_path)

        output_video = video_path.replace(".mp4", "_subtitled.mp4")

        st.info("Burning subtitles into video...")

        subprocess.run([
            "ffmpeg",
            "-i", video_path,
            "-vf", f"subtitles={srt_path}",
            "-c:a", "copy",
            output_video
        ])

        st.success("Done!")

        st.video(output_video)

        with open(srt_path, "rb") as f:
            st.download_button("Download SRT", f, file_name="subtitles.srt")
