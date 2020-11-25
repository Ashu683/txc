from flask import Flask, request, render_template, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import json
import warnings
warnings.filterwarnings("ignore")
import h5py    
import numpy as np 

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])

def index():
    print('at 1')
    return render_template('index2.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    print("At 2")
    data = request.form.get('search')
    hf = h5py.File('txc.h5', 'r')
    model = hf.get('train.py')
    max_vocab_size = 10000
    tokenizer = Tokenizer(num_words=max_vocab_size)
    tokenizer.fit_on_texts(data)
    seq = tokenizer.texts_to_sequences([data])     

    Data = preprocessing.sequence.pad_sequences(sequences=seq, maxlen=20)

    prob = model.predict_class(data)
    print("PROBABILITY::",prob)
    if prob[0][0] > str(0.5):
        sentiment = "Positive"
        return render_template('index2.html',pred='Your Comment is {}'.format(sentiment))
    else:
        sentiment = "Negative"
        return render_template('index2.html',pred='Your Comment is {}'.format(sentiment))

    
    
if __name__ == '__main__':

    app.run(debug=True, port=5000)
