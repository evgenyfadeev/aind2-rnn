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
        # append the last element following the window
        y.append(series[idx])
        # append the elements of the window
        x.append(series[idx-window_size:idx])
        # increment the window offset by the step size
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
    # get all chars in text
    chars = set(text)
    # replace individual accented chars
    text = text.replace('à', 'a')
    text = text.replace('â', 'a')
    text = text.replace('è', 'e')
    text = text.replace('é', 'e')
    for char in chars:
        # keep letters
        if char in string.ascii_letters:
            continue
        #if char in string.digits:
        #    continue

        # keep allowed punctuation chars
        if char in punctuation:
            continue

        # all remaining chars replace with an empty space
        text = text.replace(char, ' ')

    # replace repeating spaces with one at each position
    return re.sub('\s+', ' ', text).strip()

def window_transform_text(series, window_size, step_size):
    # use the series transform function
    x, y = window_transform_series(series, window_size, step_size)
    # convert results to python lists
    return x.tolist(), y.tolist()

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
