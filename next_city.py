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
    
    print("X.shape: {} & Y.shape: {}".format(X.shape, Y.shape))
    print("\t\t\t ----- use user country for prediction? \t {} ------- \t\t\t".format(user_ucountry))
    
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
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
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
    #start_index = random.randint(0, len(text) - Tx - 1)
    
    #generated = ''
    #sentence = text[start_index: start_index + Tx]
    #sentence = '0'*Tx
    #usr_input = input("Write the beginning of your poem, the Shakespearian machine will complete it.")
    # zero pad the sentence to Tx characters.
    #sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    #generated += sentence
#
    #sys.stdout.write(usr_input)

    #for i in range(400):
"""
        #x_pred = np.zeros((1, Tx, len(chars)))
        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature = 1.0)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
        
        if next_char == '\n':
            continue
        
    # Stop at the end of a line (4 lines)
    print()
    
print("Loading text data...")
text = io.open('shakespeare.txt', encoding='utf-8').read().lower()
#print('corpus length:', len(text))

Tx = 40
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
#print('number of unique characters in the corpus:', len(chars))

print("Creating training set...")
X, Y = build_data(text, Tx, stride = 3)
print("Vectorizing training set...")
x, y = vectorization(X, Y, n_x = len(chars), char_indices = char_indices) 
print("Loading model...")
model = load_model('models/model_shakespeare_kiank_350_epoch.h5')

"""
def generate_output():
    generated = ''
    #sentence = text[start_index: start_index + Tx]
    #sentence = '0'*Tx
    usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
    # zero pad the sentence to Tx characters.
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    generated += usr_input 

    sys.stdout.write("\n\nHere is your poem: \n\n") 
    sys.stdout.write(usr_input)
    for i in range(400):

        x_pred = np.zeros((1, Tx, len(chars)))

        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, chars, temperature = 1.0)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

        if next_char == '\n':
            continue

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