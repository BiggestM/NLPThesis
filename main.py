import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from flask import Flask, request, render_template
import os

app = Flask(__name__)

# Load and preprocess the data
tokenizer = Tokenizer()

with open('poems.txt', 'r', encoding='utf-8') as file:
    data = file.read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Load the pre-trained model
model = tf.keras.models.load_model("poem_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_poem', methods=['POST'])
def generate_poem():
    seed_text = request.json['seed_text']
    generated_text = generate_text(seed_text)
    return generated_text

def generate_text(seed_text):
    next_words = 100
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

if __name__ == '__main__':
    app.run(debug=True)
