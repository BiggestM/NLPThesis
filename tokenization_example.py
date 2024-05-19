import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define some sample sentences
sentences = [
    'I love my cats',
    'I love parrots',
    'You love cats!',
    'Do you like petting cats?'
]

# Create a tokenizer object with a vocabulary size of 100 words and an out-of-vocabulary token
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

# Fit tokenizer on the sentences to create a word index
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index  # Get the word index

# Convert sentences to sequences of token indices
sequences = tokenizer.texts_to_sequences(sentences)

# Pad sequences to ensure uniform length
padded = pad_sequences(sequences, maxlen=4)

# Print word index, sequences, and padded sequences
print(word_index)
print("\n", sequences)
print("\n", padded)

# Define some test data
test_data = [
    'I really love cats',
    'my cat loves my bird'
]

# Convert test data to sequences using the same tokenizer
test_seq = tokenizer.texts_to_sequences(test_data)

# Pad test sequences
padded = pad_sequences(test_seq, maxlen=10)

# Print padded test sequences
print("\n", test_seq)
print("\n", padded)
