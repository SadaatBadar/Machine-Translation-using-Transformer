import streamlit as st
import subprocess
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Machine Translation Project", layout="wide")
st.title("🌐 Machine Translation Using Transformers")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    return whisper, tokenizer, translator

whisper_model, tokenizer, translator = load_models()

# ================= TRANSLATION =================
def translate_to_hindi(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    outputs = translator.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================= SRT GENERATION =================
def generate_srt(segments):
    with open("subs.srt", "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):

            def fmt(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"

            en = seg.text.strip()
            hi = translate_to_hindi(en)

            f.write(f"{i}\n")
            f.write(f"{fmt(seg.start)} --> {fmt(seg.end)}\n")
            f.write(f"{hi}\n\n")

# ================= UI TABS =================
tab1, tab2 = st.tabs(["📝 Text → Hindi", "🎬 Video → Hindi Captions"])

# =====================================================
# TAB 1: TEXT TRANSLATION
# =====================================================
with tab1:
    st.subheader("English → Hindi Translator")

    user_text = st.text_area("Enter English text")

    if st.button("Translate Text"):
        if user_text.strip():
            hindi = translate_to_hindi(user_text)
            st.success("Translation:")
            st.write(hindi)
        else:
            st.warning("Please enter some text")

# =====================================================
# TAB 2: VIDEO CAPTIONS (ON VIDEO)
# =====================================================
with tab2:
    st.subheader("Video → Hindi Captions (Burned into Video)")

    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "mkv"],
        key="video"
    )

    if video_file:
        with open("input.mp4", "wb") as f:
            f.write(video_file.read())

        st.video("input.mp4")

        if st.button("Generate Captions"):
            with st.spinner("Extracting audio..."):
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", "input.mp4",
                        "-ar", "16000",
                        "-ac", "1",
                        "audio.wav"
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            with st.spinner("Transcribing & Translating..."):
                segments, info = whisper_model.transcribe("audio.wav")

            generate_srt(segments)

            with st.spinner("Burning captions into video..."):
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", "input.mp4",
                        "-vf", "subtitles=subs.srt",
                        "output.mp4"
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            st.success(f"Detected language: {info.language}")
            st.video("output.mp4")
