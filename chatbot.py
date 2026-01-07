
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import nltk
import string
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# -----------------------------
# Downloads NLTK (if needed)
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# Stop words
# -----------------------------
stop_words = set(stopwords.words('english'))


# -----------------------------
# Load text
# -----------------------------
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    return sent_tokenize(text, language="english")


# -----------------------------
# Preprocess
# -----------------------------
def preprocess(sentence):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens


# -----------------------------
# Similarity
# -----------------------------
def similarity(sentence1, sentence2):
    s1 = set(preprocess(sentence1))
    s2 = set(preprocess(sentence2))
    if not s1 or not s2:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))


# -----------------------------
# Most relevant sentence
# -----------------------------
def get_most_relevant_sentence(user_query, sentences):
    max_similarity = 0
    best_sentence = "Sorry, I don't understand your question."
    for sentence in sentences:
        score = similarity(user_query, sentence)
        if score > max_similarity:
            max_similarity = score
            best_sentence = sentence
    return best_sentence


# -----------------------------
# Chatbot
# -----------------------------
def chatbot(user_input, sentences):
    return get_most_relevant_sentence(user_input, sentences)


# -----------------------------
# Streamlit UIX
# -----------------------------
def main():
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

    # Title
    st.markdown("<h1 style='text-align:center; color:#4B0082;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Ask any question about Artificial Intelligence</p>",
                unsafe_allow_html=True)

    # Load sentences
    sentences = load_text("ia.txt")

    # Input box with nice layout
    user_input = st.text_area("Type your question here:", height=80)

    # Button in the same line using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Ask Chatbot"):
            if user_input.strip() != "":
                response = chatbot(user_input, sentences)
                st.markdown("<h3 style='color:#008000;'>🤖 Response:</h3>", unsafe_allow_html=True)
                st.info(response)
            else:
                st.warning("Please type a question first!")

    # Optional: add sample questions
    st.markdown("---")
    st.markdown("💡 **Example Questions:**")
    st.markdown("- What is artificial intelligence?")
    st.markdown("- Where is AI used?")
    st.markdown("- What is machine learning?")
    st.markdown("- Define NLP")


if __name__ == "__main__":
    main()
