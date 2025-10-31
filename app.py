import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load model (English → Hindi)
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

#  UI Design 
st.set_page_config(page_title="Machine Translation using Transformer", page_icon="🌍", layout="centered")

# Custom CSS for background & styling
st.markdown("""
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #1f1c2c, #928DAB);
        color: white;
    }

    /* Title */
    .big-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #FFD700;
        text-shadow: 2px 2px 8px #000;
    }

    /* Text area */
    textarea {
        background-color: #2E2E3A !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid #FFD700 !important;
    }

    /* Button */
    div.stButton > button {
        background-color: #FFD700;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #FFA500;
        color: white;
        transform: scale(1.05);
    }

    /* Translation box */
    .translation-box {
        font-size: 22px;
        background: rgba(255,255,255,0.15);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #FFD700;
        backdrop-filter: blur(10px);
        margin-top: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 14px;
        margin-top: 40px;
        color: #CCCCCC;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='big-title'>🌍 Machine Translation using Transformer</h1>", unsafe_allow_html=True)
st.write("### Translate English text into Hindi in real-time using a Transformer-based MarianMT model.")

# Input area
english_text = st.text_area("✍️ Enter English text here:", "", height=150, placeholder="Type something like: 'How are you today?'")

# Translate button
if st.button("🚀 Translate"):
    if english_text.strip():
        inputs = tokenizer(english_text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs, max_length=100)
        hindi_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        st.success("✅ Translation Successful!")
        st.markdown(f"<div class='translation-box'>{hindi_text}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some English text above to translate.")

# Footer
st.markdown("<div class='footer'>✨ Made with ❤️ using Hugging Face Transformers & Streamlit ✨</div>", unsafe_allow_html=True)
