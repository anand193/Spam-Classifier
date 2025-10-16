import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ✅ Ensure nltk data is stored locally (fixes LookupError)
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

ps = PorterStemmer()
stop_words = stopwords.words('english')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return ' '.join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title('📩 Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.error('🚨 Spam')
    else:
        st.success('✅ Not Spam')

