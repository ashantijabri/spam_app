import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


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