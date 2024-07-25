from flask import Flask, request, jsonify, render_template
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
import re

app = Flask(__name__)

# Load the corpus text with fallback encoding
def load_corpus(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

corpus = load_corpus('corpus.txt')


corpus_sentences = sent_tokenize(corpus)

model = SentenceTransformer('paraphrase-mpnet-base-v2')


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = text.strip()
    return text


corpus_sentences = [clean_text(sentence) for sentence in corpus_sentences]

corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True)

conversation_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = generate_response(user_message)
    return jsonify({'response': response})

def generate_response(user_message):
    global conversation_history
    cleaned_message = clean_text(user_message)
    conversation_history.append({"role": "user", "content": cleaned_message})

    best_match = find_best_match(cleaned_message)

    if best_match:
        response = extract_answer(user_message, best_match)
        conversation_history.append({"role": "assistant", "content": response})
    else:
        response = "I'm sorry, I couldn't find an answer to your question. Please try rephrasing it or ask something else."
        conversation_history.append({"role": "assistant", "content": response})
    
    return response

def find_best_match(cleaned_message):
    best_match = None
    highest_similarity = 0.0
    user_embedding = model.encode(cleaned_message, convert_to_tensor=True)


    similarities = util.pytorch_cos_sim(user_embedding, corpus_embeddings).flatten()


    highest_similarity = similarities.max().item()
    best_match_index = similarities.argmax().item()

    if highest_similarity >= 0.6: 
        best_match = corpus_sentences[best_match_index]

    print(f"User message: {cleaned_message}")
    print(f"Best match: {best_match} with similarity {highest_similarity}")

    return best_match

def extract_answer(user_message, best_match):
 
    match_index = corpus_sentences.index(best_match)
    
 
    start = max(0, match_index - 2)
    end = min(len(corpus_sentences), match_index + 3)
    context = " ".join(corpus_sentences[start:end])
    
    response = f"{context}"
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
