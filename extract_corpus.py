import fitz  # 
import json

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

pdf_path = 'Corpus.pdf'
corpus_text = extract_text_from_pdf(pdf_path)

with open('corpus.txt', 'w') as file:
    file.write(corpus_text)

print("Corpus extracted and saved to corpus.txt")
