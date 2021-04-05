import tensorflow as tf
import keras
import streamlit as st
import re
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding

class_ids = ['spam', 'not spam']

st.write ('# Ashanti\'s Spam Detection App')

message_text = st.text_input("Enter a message for spam evaluation")


@st.cache
def spam_check(sms):
  sms=[sms]
  sms_proc = tokenizer.texts_to_sequences(sms)
  sms_proc = pad_sequences(sms_proc, maxlen=max_length, padding='post')
  pred = (model.predict(sms_proc) > 0.5).astype("int32").item()
  print(pred)