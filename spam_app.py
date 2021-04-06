import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
import streamlit as st
import re
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
import pickle
from keras import backend as K
import base64
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache(allow_output_mutation=True)
def load_model():
    model = load_model('spam_model')
    model._make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    return model

max_length = 10

st.write ('# Spam Detection App')

message_text = st.text_input("Enter a text message or email for spam evaluation")

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = tf.keras.models.load_model('spam_model')

def spam_check(model, sms):
  sms=[sms]
  sms_proc = tokenizer.texts_to_sequences(sms)
  sms_proc = pad_sequences(sms_proc, maxlen=max_length, padding='post')
  pred = (model.predict(sms_proc) > 0.5).astype("int32").item()
  if pred == 1:
    pred = 'This is Spam'
  else:
    pred = 'This is not Spam'
  return pred

if message_text != '':
    with st.spinner('Generating explanations'):
        result = spam_check(model, message_text)
        st.write(result)