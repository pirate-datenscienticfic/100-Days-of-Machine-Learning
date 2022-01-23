import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import streamlit as st
import re
from tensorflow import keras



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

VOCAB_SIZE = 1000

## Loading in Model
# # json_file = open('G:\Assignments\100DaysofML\100-Days-of-Machine-Learning\Streamlit-Deployment\LSTM_model\model.json', 'r')
# # # json_file = open('model.json', 'r')
# # loaded_model_json = json_file.read()
# # json_file.close()

# # with open("G:\Assignments\100DaysofML\100-Days-of-Machine-Learning\Streamlit-Deployment\LSTM_model\model.json", 'r') as json_file:
# with open("model.json", 'r') as json_file:
#     loaded_model_json = json_file.read()
# lstm_model = model_from_json(loaded_model_json)

# # lstm_model.load_weights("G:\Assignments\100DaysofML\100-Days-of-Machine-Learning\Streamlit-Deployment\LSTM_model\model.h5")
# lstm_model.load_weights("model.h5")
lstm_model = keras.models.load_model('model/')



## Prediction function
def sentiment_prediction(review):
    sentiment = []
    input_review = review
    # input_review = [x.lower() for x in input_review]
    # input_review = [re.sub('[^a-zA-z0-9\s]', '', x) for x in input_review]
    sentiment = lstm_model.predict(np.array([input_review]))
    
    if (sentiment > 0.0):
        pred = "Positive"
    else:
        pred = "Negative"
    
    return pred


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
    
