import re
import string
import numpy as np
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size, step_size=1):
    """We can use just one function to window the data
    by using the step size parameter"""
    idx = window_size
    y = list()
    x = list()
    while idx < len(series):
        y.append(series[idx])
        x.append(series[idx-window_size:idx])
        idx += step_size

    return np.array(x), np.array(y)


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    chars = set(list(text))
    text = text.replace('à', 'a')
    text = text.replace('â', 'a')
    text = text.replace('è', 'e')
    text = text.replace('é', 'e')
    for char in chars:
        if char in string.ascii_letters:
            continue
        if char in string.digits:
            continue
        if char in punctuation:
            continue
        text = text.replace(char, ' ')
    return re.sub('\s+', ' ', text).strip()

window_transform_text = window_transform_series

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
