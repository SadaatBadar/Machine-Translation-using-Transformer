```python
import streamlit as st
import tempfile
import subprocess
import os
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import torch

st.set_page_config(page_title="HinSync", layout="wide")

# ======================
# FONT PATH FIX
# ======================

FONT_PATH = os.path.join("fonts", "NotoSansDevanagari.ttf")

# ======================
# LOAD MODELS
# ======================

@st.cache_resource
def load_whisper():
    return WhisperModel("medium", device="cpu", compute_type="int8")

@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

whisper_model = load_whisper()
tokenizer, translator_model = load_translator()

# ======================
# AUDIO PREPROCESSING
# ======================

def extract_clean_audio(video_path):
    clean_audio = video_path.replace(".mp4", "_clean.wav")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-af", "loudnorm,afftdn",
        clean_audio,
        "-y"
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return clean_audio

# ======================
# TRANSLATION FUNCTIONS
# ======================

def clean_text(text):
    fillers = ["uh", "um", "you know", "like"]
    for f in fillers:
        text = text.replace(f, "")
    return text.strip()

def translate_text(text):
    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        translated = translator_model.generate(**inputs, num_beams=4)

    return tokenizer.decode(translated[0], skip_special_tokens=True)

# ======================
# TIMESTAMP FORMAT
# ======================

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

# ======================
# CREATE SRT
# ======================

def generate_srt(segments, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):

            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)

            text = seg.text.strip()

            # Split long segments for better translation
            if len(text.split()) > 20:
                parts = text.split(".")
                hindi = " ".join([translate_text(p) for p in parts if p.strip()])
            else:
                hindi = translate_text(text)

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{hindi}\n\n")

# ======================
# BURN SUBTITLES
# ======================

def burn_subtitles(video_path, srt_path, output_path):

    font_abs = os.path.abspath(FONT_PATH)

    vf = (
        f"subtitles='{srt_path}':"
        f"fontsdir='{os.path.dirname(font_abs)}':"
        f"force_style='FontName=Noto Sans Devanagari'"
    )

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path,
        "-y"
    ]

    subprocess.run(command, check=True)

# ======================
# UI
# ======================

st.title("🎬 HinSync - English Speech → Hindi Subtitles")

tabs = st.tabs(["🎥 Subtitle Generator", "🌐 Text Translator"])

# =================================================
# TAB 1 - VIDEO
# =================================================

with tabs[0]:

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv"])

    if uploaded_file:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        st.video(video_path)

        if st.button("Generate Subtitles"):

            with st.spinner("🔊 Extracting & cleaning audio..."):
                clean_audio = extract_clean_audio(video_path)

            with st.spinner("🧠 Transcribing..."):
                segments, _ = whisper_model.transcribe(
                    clean_audio,
                    beam_size=5,
                    best_of=5,
                    temperature=0
                )
                segments = list(segments)

            with st.spinner("🌐 Creating Hindi SRT..."):
                srt_path = video_path.replace(".mp4", ".srt")
                generate_srt(segments, srt_path)

            with st.spinner("🎬 Burning subtitles into video..."):
                output_video = video_path.replace(".mp4", "_subtitled.mp4")
                burn_subtitles(video_path, srt_path, output_video)

            st.success("✅ Done!")

            with open(output_video, "rb") as f:
                st.download_button(
                    "Download Video",
                    f,
                    file_name="hinsync_output.mp4"
                )

# =================================================
# TAB 2 - TEXT TRANSLATOR
# =================================================

with tabs[1]:

    user_text = st.text_area("Enter English Text")

    if st.button("Translate Text"):

        if user_text.strip():

            with st.spinner("Translating..."):
                result = translate_text(user_text)

            st.subheader("Hindi Output")
            st.write(result)
```
