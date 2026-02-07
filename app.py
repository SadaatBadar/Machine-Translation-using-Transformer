import streamlit as st
import subprocess
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

# ================= PAGE CONFIG =================
st.set_page_config(page_title="HinSync", layout="wide")
st.title("🎧 HinSync")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    whisper = WhisperModel("base", device="cpu", compute_type="int8")

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

    return whisper, tokenizer, translator


whisper_model, tokenizer, translator = load_models()

# ================= BATCH TRANSLATION =================
def translate_batch(texts):
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    translated = translator.generate(**batch)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

# ================= SRT GENERATION =================
def generate_srt(segments):

    texts = [seg.text.strip() for seg in segments]
    translations = translate_batch(texts)

    with open("subs.srt", "w", encoding="utf-8") as f:

        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        for i, (seg, hi) in enumerate(zip(segments, translations), 1):

            f.write(f"{i}\n")
            f.write(f"{fmt(seg.start)} --> {fmt(seg.end)}\n")
            f.write(f"{hi}\n\n")

# ================= CREATE TABS =================
tab1, tab2 = st.tabs(["📝 Text → Hindi", "🎬 Video → Hindi Captions"])

# =====================================================
# TAB 1 → TEXT TRANSLATOR
# =====================================================
with tab1:

    st.subheader("English → Hindi Translator")

    user_text = st.text_area("Enter English text")

    if st.button("Translate Text"):

        if user_text.strip():

            translation = translate_batch([user_text])[0]

            st.success("Translation:")
            st.write(translation)

        else:
            st.warning("Please enter some text")

# =====================================================
# TAB 2 → VIDEO SUBTITLES
# =====================================================
with tab2:

    st.subheader("Video → Hindi Captions")

    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "mkv"]
    )

    if video_file:

        with open("input.mp4", "wb") as f:
            f.write(video_file.read())

        st.video("input.mp4")

        if st.button("Generate Captions"):

            # Extract audio
            with st.spinner("Extracting audio..."):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", "input.mp4", "-ar", "16000", "-ac", "1", "audio.wav"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            # Whisper transcription
            with st.spinner("Transcribing speech..."):
                segments, info = whisper_model.transcribe(
                    "audio.wav",
                    beam_size=1
                )

                segments = list(segments)

            # Subtitle generation
            with st.spinner("Translating & generating subtitles..."):
                generate_srt(segments)

            # Burn subtitles
            with st.spinner("Burning subtitles into video..."):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", "input.mp4", "-vf", "subtitles=subs.srt", "output.mp4"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            st.success(f"Detected language: {info.language}")
            st.video("output.mp4")
