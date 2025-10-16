import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------------
# 🔹 NLTK setup (deployment-safe)
# ---------------------------
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Download punkt and stopwords to local folder
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

# Add local folder to nltk search path
nltk.data.path.append(nltk_data_dir)

# ---------------------------
# 🔹 Preprocessing setup
# ---------------------------
ps = PorterStemmer()
stop_words = stopwords.words('english')

def transform_text(text):
    # 1️⃣ Lowercase
    text = text.lower()
    # 2️⃣ Tokenize
    text = nltk.word_tokenize(text)
    
    # 3️⃣ Remove non-alphanumeric tokens
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()
    
    # 4️⃣ Remove stopwords and punctuation
    y = [i for i in text if i not in stop_words and i not in string.punctuation]
    text = y[:]
    y.clear()
    
    # 5️⃣ Stemming
    y = [ps.stem(i) for i in text]
    
    return ' '.join(y)

# ---------------------------
# 🔹 Load model and vectorizer
# ---------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------------------
# 🔹 Streamlit UI
# ---------------------------
st.title('📩 Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess input
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        
        # Display result
        if result == 1:
            st.error('🚨 Spam')
        else:
            st.success('✅ Not Spam')


