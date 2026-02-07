import streamlit as st
import subprocess
import os
import tempfile
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="HinSync", page_icon="🎬")

FONT_PATH = "fonts/NotoSansDevanagari.ttf"

# Load Whisper (smaller model = faster)
@st.cache_resource
def load_whisper():
    return WhisperModel("base", compute_type="int8")

# Load Translation Model
@st.cache_resource
def load_translation():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

whisper_model = load_whisper()
tokenizer, translation_model = load_translation()

# ---------------- UI ---------------- #

st.title("🎬 HinSync")
st.write("Upload video → English Speech → Hindi Subtitles")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

# ---------------- FUNCTIONS ---------------- #

def transcribe_audio(video_path):
    segments, _ = whisper_model.transcribe(video_path, beam_size=1)
    return list(segments)

def translate_text(texts):
    batch = tokenizer(texts, return_tensors="pt", padding=True)
    translated = translation_model.generate(**batch)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

def create_srt(segments, translations, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (seg, trans) in enumerate(zip(segments, translations)):
            f.write(f"{i+1}\n")
            f.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
            f.write(f"{trans}\n\n")

def burn_subtitles(video_path, srt_path, output_path):

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}:force_style='FontName=Noto Sans Devanagari'",
        "-c:a", "copy",
        output_path,
        "-y"
    ]

    subprocess.run(command)

# ---------------- MAIN PROCESS ---------------- #

if uploaded_video:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name

    st.info("🎙 Transcribing Audio...")
    segments = transcribe_audio(video_path)

    texts = [seg.text for seg in segments]

    st.info("🌐 Translating to Hindi...")
    translations = translate_text(texts)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as temp_srt:
        srt_path = temp_srt.name

    create_srt(segments, translations, srt_path)

    output_video = video_path.replace(".mp4", "_subtitled.mp4")

    st.info("🔥 Burning Subtitles...")
    burn_subtitles(video_path, srt_path, output_video)

    st.success("✅ Done!")

    st.video(output_video)

    with open(output_video, "rb") as f:
        st.download_button("⬇ Download Video", f, file_name="Hindi_Subtitled.mp4")
