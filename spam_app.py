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

st.text_input("Enter a message for spam evaluation")