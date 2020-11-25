import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Embedding


train_df = pd.read_csv('train.csv')

rowsums=train_df.iloc[:,2:].sum(axis=1)
train_df['clean']=(rowsums==0)

# Input Data
train_texts = train_df['comment_text']
# Output Label
train_labels = train_df['clean']

from tensorflow.keras.preprocessing.text import Tokenizer
# set size of vocabulary
# To Do: try different size 
max_vocab_size = 10000
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
print(sequences[0])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

from tensorflow.keras import preprocessing
training_sequences = sequences[:10000]
training_labels = train_labels[:10000]
seq_max_len = 20
# training padded sequences
train_seq_pad = preprocessing.sequence.pad_sequences(sequences=training_sequences, maxlen=seq_max_len)

# testing padded sequences
testing_sequences = sequences[10000:11000]
testing_labels = train_labels[10000:11000]
test_seq_pad = preprocessing.sequence.pad_sequences(sequences=testing_sequences, maxlen=seq_max_len)


model_1 = Sequential()

# no. of unique words in the text data, each word in vocab will be assigned an index (dimension).
# = max_vocab_size defined above
vocab_size = 10000 
seq_max_len = 20 

# dimension of word embedding model (output dimension of embedding layer)pip 
embedding_dim = 8 

## layer 1: add Embedding Layer in the network
 
model_1.add(Embedding(vocab_size, embedding_dim, input_length=seq_max_len))
 

## layer 2: flatten the input of shape [batch_size, embedding_dim, seq_max_len] 
model_1.add(Flatten())
 
## layer 3 (final/output layer): Dense layer 
 
model_1.add(Dense(1, activation='sigmoid'))
 
model_1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 

model_1.fit(train_seq_pad, np.asarray(training_labels), epochs=10, batch_size=100, validation_split=0.2)

model_1.save("/third/txc.h5")