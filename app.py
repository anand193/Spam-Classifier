import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --------------------------
# NLTK downloads (Streamlit safe)
# --------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --------------------------
# Text preprocessing
# --------------------------
ps = PorterStemmer()
stop_words = stopwords.words('english')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    y = [i for i in text if i not in stop_words and i not in string.punctuation]
    text = y[:]
    y.clear()

    # Stemming
    y = [ps.stem(i) for i in text]

    return ' '.join(y)

# --------------------------
# Load saved model and vectorizer
# --------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --------------------------
# Streamlit App
# --------------------------
st.title('📩 Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error('🚨 Spam')
        else:
            st.success('✅ Not Spam')


