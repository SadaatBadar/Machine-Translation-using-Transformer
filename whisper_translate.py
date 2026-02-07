from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import torch

# ---------- LOAD MODELS ----------

# Whisper (speech → text)
whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# Translator (English → Hindi)
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator = MarianMTModel.from_pretrained(model_name)

# ---------- TRANSLATION FUNCTION ----------

def translate_to_hindi(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    outputs = translator.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- TRANSCRIBE + TRANSLATE ----------

segments, info = whisper_model.transcribe("audio.wav")

print("Detected language:", info.language)
print("-" * 50)

for segment in segments:
    english = segment.text.strip()
    hindi = translate_to_hindi(english)

    print(f"[{segment.start:.2f}s → {segment.end:.2f}s]")
    print("EN:", english)
    print("HI:", hindi)
    print("-" * 50)
