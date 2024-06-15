import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import streamlit as st
import numpy as np
st.text_input("Enter your text here", key="text")
st.text_input("Number of words you want to predict",key="num")

# You can access the value at any point with:

file_path = 'Data.txt'
with open(file_path, 'r') as file:
    file_content = file.read()
faqs=file_content
tokenizer = Tokenizer()

tokenizer.fit_on_texts([faqs])  # assigns index to each word
num_classes = len(tokenizer.word_index)  # to print the index of each word
input_sequences = []
for sentence in faqs.split('.'):
    # as it became a 2d array to print as multiple 1d arrays
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i+1])
# need to do  padding
max_len = max([len(x) for x in input_sequences])
padded_input_sequences = pad_sequences(
    input_sequences, maxlen=max_len, padding='pre')

X = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]
# classification starts from zero but our indices for the words are from one so we added one
y = to_categorical(y, num_classes+1)
# this does one hot encoding for y

model = Sequential()
# every sentences every word is converted to 100 numbers
model.add(Embedding(num_classes+1, 100, input_length=max_len))
model.add(LSTM(150))
model.add(Dense(num_classes+1, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# model.summary()
model.fit(X, y, epochs=100)
text = st.session_state.text
num_words=st.session_state.num
for i in range(int(num_words)):
    # tokenize
    token_text = tokenizer.texts_to_sequences([text])[0]
    # padding
    padded_token_text = pad_sequences(
        [token_text], maxlen=max_len, padding='pre')
    # predict
    pos = np.argmax(model.predict(padded_token_text))  # (gives 1,283 probs)
    for word, index in tokenizer.word_index.items():
        if index == pos:
            text = text+" "+word
st.write(text)        
