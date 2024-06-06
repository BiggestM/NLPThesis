import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the data
# Initialize the tokenizer
tokenizer = Tokenizer()

# Read the text data from the file
with open('poems.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Convert text to lowercase and split into lines
corpus = data.lower().split("\n")

# Fit the tokenizer on the text corpus
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # Add 1 for padding

# Create input sequences using n-grams
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad the sequences to ensure uniform length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split the data into input sequences (xs) and labels (ys)
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Define and train the model
# Initialize the Sequential model
model = Sequential()
# Add an Embedding layer
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
# Add a Bidirectional LSTM layer with return_sequences=True
model.add(Bidirectional(LSTM(150, return_sequences=True)))
# Add a Dropout layer to prevent overfitting
model.add(Dropout(0.2))
# Add another Bidirectional LSTM layer
model.add(Bidirectional(LSTM(150)))
# Add a Dense output layer with softmax activation
model.add(Dense(total_words, activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss
adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the model and save the training history
history = model.fit(xs, ys, epochs=100, verbose=1)

# Save the trained model (optional)
# model.save("poem_model.h5")

# Function to plot accuracy graph
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title(f'Model {string} over Epochs')
    plt.show()

# Plot the accuracy graph
plot_graphs(history, 'accuracy')

# Display the final accuracy in percentage
final_accuracy = history.history['accuracy'][-1] * 100
print(f"Final training accuracy: {final_accuracy:.2f}%")

# Generate text using the trained model
seed_text = "Looks it not like the king?  Verily, we must go! "
next_words = 100

# Generate the next 100 words based on the seed text
for _ in range(next_words):
    # Tokenize and pad the seed text
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    # Predict the next word
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
    output_word = ""
    # Find the predicted word in the tokenizer's word index
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    # Append the predicted word to the seed text
    seed_text += " " + output_word

# Print the generated text
print(seed_text)
