import streamlit as st
import tempfile
import subprocess
import os
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import torch

st.set_page_config(page_title="HinSync", layout="wide")

# =========================
# LOAD MODELS (CACHED)
# =========================

@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8")

@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

whisper_model = load_whisper()
tokenizer, translator_model = load_translator()

# =========================
# TRANSLATION FUNCTION
# =========================

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        translated = translator_model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# =========================
# SRT FORMATTER
# =========================

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

# =========================
# CREATE SRT FILE
# =========================

def generate_srt(segments, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)

            hindi = translate_text(seg.text.strip())

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{hindi}\n\n")

# =========================
# BURN SUBTITLES
# =========================

def burn_subtitles(video_path, srt_path, output_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}:force_style='FontName=Noto Sans Devanagari'",
        "-c:a", "copy",
        output_path
    ]

    subprocess.run(command, check=True)

# =========================
# UI TITLE
# =========================

st.title("🎬 HinSync - Speech → Hindi Subtitles + Translation")

tabs = st.tabs(["🎥 Subtitle Generator", "🌐 Text Translator"])

# ======================================================
# TAB 1 : VIDEO SUBTITLE GENERATOR
# ======================================================

with tabs[0]:

    st.header("Upload Video for Hindi Subtitles")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv"])

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        st.video(video_path)

        if st.button("Generate Subtitles"):

            with st.spinner("Transcribing..."):

                segments, _ = whisper_model.transcribe(video_path)

                segments = list(segments)

            with st.spinner("Generating Hindi SRT..."):

                srt_path = video_path.replace(".mp4", ".srt")
                generate_srt(segments, srt_path)

            with st.spinner("Burning Subtitles into Video..."):

                output_video = video_path.replace(".mp4", "_subtitled.mp4")

                burn_subtitles(video_path, srt_path, output_video)

            st.success("✅ Done!")

            with open(output_video, "rb") as f:
                st.download_button(
                    "Download Subtitled Video",
                    f,
                    file_name="hinsync_output.mp4"
                )

# ======================================================
# TAB 2 : TEXT TRANSLATOR
# ======================================================

with tabs[1]:

    st.header("English → Hindi Text Translator")

    user_text = st.text_area("Enter English Text")

    if st.button("Translate Text"):

        if user_text.strip() == "":
            st.warning("Enter text first")
        else:
            with st.spinner("Translating..."):
                result = translate_text(user_text)

            st.subheader("Hindi Translation")
            st.write(result)
