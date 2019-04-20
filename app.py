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

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper

def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    # use str instead of basestring if using Python 3.x
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    # use str instead of basestring if using Python 3.x
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

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
        token_list = [words2idx[w] for w in text.split() if w in words2idx]
        if len(token_list) == 0: token_list = [int(10000 * np.random.uniform(0, 1, 1)[0])]
        token_list = pad_sequences([token_list], maxlen=MAX_LENGTH-1, padding='pre')

        predicted = np.squeeze(qrnn_model.predict(token_list, verbose=0))
        
        output_word = idx2word[np.argmax(predicted)]
        if output_word in text.split(' ')[-5:]:
            output_word = idx2word[get_second_best_idx(predicted)]
        
        text += " " + output_word
        
    return text.strip()

def beam_search_decoding(test_text, states, model, b, h, next_words):
    global MAX_LENGTH, words2idx, idx2word
    if h == next_words:
        return (' ').join([idx2word[w] for w in states[0][0]])
    
    else:
        order = []
        for s in states:
            if len(s[0]) == 0: 
                token_list = [words2idx[w] for w in test_text.split()]
                if len(token_list) == 0: token_list = [int(10000 * np.random.uniform(0, 1, 1)[0])]
            else: token_list = s[0]
            
            distribution = np.squeeze(qrnn_model.predict(pad_sequences([token_list], maxlen=MAX_LENGTH-1, 
                                                                       padding='pre'), verbose=0))
            for idx, p in enumerate(distribution):
                if idx not in token_list[-3:]:
                    order.append((s[1] - np.log(p), token_list + [idx]))

        order = sorted(order)
        if h != 0: 
            order = [(v, k) for (k, v) in order[:b]]
        else:
            new_order = []
            s = set()
            for (k, v) in order:
                if v[0] not in s:
                    new_order.append((v, k))
                    if len(new_order) == b:
                        break
                    s.add(v[0])
            order = new_order
            
        return beam_search_decoding(test_text, order, model, b, h+1, next_words)


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

@crossdomain(origin='*')
@app.route('/generateDocument', methods=['GET'])
def generateDocument():
    global st
    initial_text = str(request.args.get('text'))
    if not request.args.get('number_next_words'): number_next_words = 40
    else: number_next_words = int(request.args.get('number_next_words'))
    
    if initial_text == '': return ''
    initial_text = st.predict(initial_text)
    
    if number_next_words == 0: return initial_text
    else:
        # return generate_text(initial_text, number_next_words, qrnn_model)
        return beam_search_decoding(initial_text, [[list(), 0] for i in range(3)], qrnn_model, 3, 0, number_next_words)

app.run(host='0.0.0.0', port=80)