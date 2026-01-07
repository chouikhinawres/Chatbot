import nltk
import string
import streamlit as st
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Downloads (first time only)
nltk.download('punkt')
nltk.download('stopwords')

# Stop words
stop_words = set(stopwords.words('english'))

# Load text
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    return sent_tokenize(text)

# Preprocess
def preprocess(sentence):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return tokens

# Similarity
def similarity(sentence1, sentence2):
    s1 = set(preprocess(sentence1))
    s2 = set(preprocess(sentence2))
    if not s1 or not s2:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))

# Best sentence
def get_most_relevant_sentence(user_query, sentences):
    max_similarity = 0
    best_sentence = "Sorry, I don't understand your question."
    for sentence in sentences:
        score = similarity(user_query, sentence)
        if score > max_similarity:
            max_similarity = score
            best_sentence = sentence
    return best_sentence

# Chatbot
def chatbot(user_input, sentences):
    return get_most_relevant_sentence(user_input, sentences)

# Streamlit App
def main():
    st.title("🤖 AI Chatbot (NLTK)")
    st.write("Ask a question about Artificial Intelligence")

    sentences = load_text("ia.txt")
    user_input = st.text_input("Your question:")

    if user_input:
        response = chatbot(user_input, sentences)
        st.success(response)

if __name__ == "__main__":
    main()
