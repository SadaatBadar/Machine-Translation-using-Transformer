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
    whisper = WhisperModel("base", device="cpu")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    return whisper, tokenizer, translator

whisper_model, tokenizer, translator = load_models()

# ================= TRANSLATION =================
def translate_to_hindi(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    outputs = translator.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================= TIME FORMAT =================
def fmt_srt(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def fmt_ass(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int((t - int(t)) * 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"

# ================= SUBTITLE GENERATION =================
def generate_subtitles(segments):

    with open("subs.srt", "w", encoding="utf-8") as srt, \
         open("subs.ass", "w", encoding="utf-8") as ass:

        # -------- ASS HEADER --------
        ass.write("[Script Info]\n")
        ass.write("ScriptType: v4.00+\n\n")

        ass.write("[V4+ Styles]\n")
        ass.write("Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n")
        ass.write("Style: Default,Noto Sans Devanagari,36,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n")

        ass.write("[Events]\n")
        ass.write("Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")

        # -------- LOOP SEGMENTS --------
        for i, seg in enumerate(segments, 1):

            en = seg.text.strip()
            hi = translate_to_hindi(en)

            # ----- SRT -----
            srt.write(f"{i}\n")
            srt.write(f"{fmt_srt(seg.start)} --> {fmt_srt(seg.end)}\n")
            srt.write(f"{hi}\n\n")

            # ----- ASS -----
            ass.write(
                f"Dialogue: 0,{fmt_ass(seg.start)},{fmt_ass(seg.end)},Default,,0,0,0,,{hi}\n"
            )

# ================= UI =================
video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv"])

if video_file:

    with open("input.mp4", "wb") as f:
        f.write(video_file.read())

    st.video("input.mp4")

    if st.button("Generate Captions"):

        with st.spinner("Extracting audio..."):
            subprocess.run(
                ["ffmpeg", "-y", "-i", "input.mp4", "-ar", "16000", "-ac", "1", "audio.wav"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        with st.spinner("Transcribing & Translating..."):
            segments, info = whisper_model.transcribe("audio.wav")

        generate_subtitles(segments)

        with st.spinner("Burning captions into video..."):
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", "input.mp4",
                    "-vf", "ass=subs.ass:fontsdir=fonts",
                    "output.mp4"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        st.success(f"Detected language: {info.language}")
        st.video("output.mp4")

        # -------- DOWNLOAD SRT --------
        with open("subs.srt", "rb") as f:
            st.download_button("Download SRT", f, file_name="subs.srt")
