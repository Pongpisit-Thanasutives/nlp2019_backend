from flask import request

from tokenizer import SertisTokenizer
import numpy as np
from utils import savefile, loadfile

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU
from keras.models import Sequential
from qrnn import *

from flask import Flask
app = Flask(__name__)

def get_second_best_idx(probs):
    return np.arange(0, probs.shape[0])[np.argsort(probs) == probs.shape[0]-2][0]

def create_model(max_sequence_len, total_words):
    global MAX_LENGTH, words2idx
    input_len = max_sequence_len - 1
    
    model = Sequential()
    model.add(Embedding(total_words, 64, input_length=input_len))
    
    model.add(QRNN(256, window_size=2, dropout=0.1, return_sequences=True,
               kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4), 
               kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(QRNN(256, window_size=2, dropout=0.1, return_sequences=False,
           kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4), 
           kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    
    model.add(Dense(total_words, activation='softmax'))
    
    return model

def generate_text(text, next_words, qrnn_model):
    global MAX_LENGTH, words2idx, idx2word

    for j in range(next_words):
        token_list = [words2idx[w] for w in text.split()]
        token_list = pad_sequences([token_list], maxlen=MAX_LENGTH-1, padding='pre')

        predicted = np.squeeze(qrnn_model.predict(token_list, verbose=0))
        
        output_word = idx2word[np.argmax(predicted)]
        if output_word in text.split(' ')[-5:]:
            output_word = idx2word[get_second_best_idx(predicted)]
        
        text += " " + output_word
        
    return text.strip()

st = SertisTokenizer()
MAX_LENGTH = 200
words2idx = loadfile('words2idx.pkl')
idx2word = loadfile('idx2word.pkl')
qrnn_model = create_model(MAX_LENGTH, len(words2idx))
qrnn_model.load_weights('QRNN_best.h5')
print(generate_text('ธนาคาร เพื่อ ประชาชน', 30, qrnn_model))

@app.route("/")
def hello():
    return "The backend server is running"

@app.route('/generateDocument', methods=['GET'])
def generateDocument():
    global st
    initial_text = str(request.args.get('text'))
    initial_text = st.predict(initial_text)
    number_next_words = int(request.args.get('number_next_words'))
    return generate_text(initial_text, number_next_words, qrnn_model)