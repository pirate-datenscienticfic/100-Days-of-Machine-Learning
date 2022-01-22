import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import streamlit as st
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



## UI
def run():
    st.title("Sentiment Analysis - LSTM Model")
    html_temp = """
    """
    st.markdown(html_temp)
    
    review = st.text_input("Enter your Review ")
    prediction = ""
    if(st.button("Predict Sentiment")):
        pass
    st.success("The Sentiment predicted by the model : {}".format(prediction))
    
    
if __name__ == "__main__":
    run()
    
