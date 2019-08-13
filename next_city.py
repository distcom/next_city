# Load Packages
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import random
import sys
import io
from sklearn.model_selection import train_test_split

def build_data(citiesSearchedSeries, Tx = 10, user_ucountry = True):
    
    # building data

                
    X = citiesSearchedSeries.apply(lambda x: x[0:-1])
    Y = citiesSearchedSeries.apply(lambda y: y[-1])
    
    if user_ucountry:
        for x in citiesSearchedSeries:
        
            t = len(x);
            if t > 2:
                for i in range(1, t-2):
                    X = X.append(pd.Series([x[0:-1-i]]), ignore_index = True)
                    Y = Y.append(pd.Series([x[-1-i]]), ignore_index = True)        
        
    else:
        for x in citiesSearchedSeries:
        
            t = len(x);
            if t > 1:
                for i in range(1, t-1):
                    X = X.append(pd.Series([x[0:-1-i]]), ignore_index = True)
                    Y = Y.append(pd.Series([x[-1-i]]), ignore_index = True)  
        

    
    print("1.1. X.shape: {} & Y.shape: {}".format(X.shape, Y.shape))
    
    X = X.reset_index(drop=True)
    X = X.apply(lambda x: max(Tx - len(x), 0) * [""] + x)
    Y = Y.reset_index(drop=True)
    
    print("2. X.shape: {} & Y.shape: {}".format(X.shape, Y.shape))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), Y_train.reset_index(drop=True), Y_test.reset_index(drop=True)


def vectorization(X, Y, n_x, citiesIndexed, Tx = 10):
    
    m = len(X)
    x = np.zeros((m, Tx, n_x), dtype=np.float32)
    y = np.zeros((m, n_x), dtype=np.float32)
    for i in range(len(X)):
        for t, city in enumerate(X[i]):
            x[i, t, citiesIndexed.index(city)] = 1
        y[i, citiesIndexed.index(Y[i])] = 1
        
    return x, y 

def create_sample(sample, n_x, citiesIndexed, Tx = 10):
    sample = [""] * (Tx - len(sample)) + sample
    x = np.zeros((1, Tx, n_x), dtype=np.float32)
    for t, city in enumerate(sample):
        x[0, t, citiesIndexed.index(city)] = 1
    
    
    return x




def city_model(input_shape, n_x):
    
    # input_shape: (m, Tx, n_x)
    
    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    city_indices = Input(shape=input_shape, dtype=np.float32) 
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(city_indices)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(n_x, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(city_indices, X)
    
    ### END CODE HERE ###
    
    return model

def sample(preds, options, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(options)), p = probas.ravel())
    return out
    #return np.argmax(probas)
    
def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    None

def predict_next_cities_(model, entries, citiesIndex, number, Tx):
    
    entries = list(filter(None, entries))
    number = max(number, 10-len(entries))
    for i in range(number): 
        smple = create_sample(entries, len(citiesIndex), citiesIndex, Tx)
        preds = model.predict(smple, verbose=0)[0]
        next_index = sample(preds, citiesIndex, temperature = 1.0)
        next_city = citiesIndex[next_index]
        if len(next_city) > 2 and next_city not in entries:
            print("{} --->>> {}".format(entries, next_city))
            entries.append(next_city)
        else:
            i -= 1