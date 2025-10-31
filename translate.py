from transformers import MarianMTModel, MarianTokenizer

# Load English → Hindi model
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_to_hindi(sentence):
    inputs = tokenizer([sentence], return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs, max_length=200)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

print("Type an English sentence (or 'quit' to exit):")
while True:
    eng = input("English: ")
    if eng.lower() == "quit":
        break
    hin = translate_to_hindi(eng)
    print("Hindi:", hin)
