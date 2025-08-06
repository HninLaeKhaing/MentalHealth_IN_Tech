import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load assets
lemmatizer = WordNetLemmatizer()
model = load_model("model/model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("model/texts.pkl", "rb"))
classes = pickle.load(open("model/labels.pkl", "rb"))

# NLP preprocessing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def chatbot_response(msg):
    intents_list = predict_class(msg)
    if intents_list:
        return get_response(intents_list, intents)
    else:
        return "I'm not sure how to respond to that. Could you please rephrase?"

# Streamlit UI
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")
st.title("💬 Mental Health Chatbot")
st.markdown("Ask me anything related to mental health. Type below and hit enter.")

# Chat input
user_input = st.text_input("You:", placeholder="Type your message here...")

# Show response
if user_input:
    response = chatbot_response(user_input)
    st.write(f"**Bot:** {response}")
