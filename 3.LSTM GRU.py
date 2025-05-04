'''
Design RNN or its variant including LSTM or GRU a) Select a suitable time series 
dataset. Example – predict sentiments based on product reviews b) Apply for prediction.
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# Step 1: Load Dataset (IMDB dataset)
max_features = 10000  # Number of words to consider in the vocabulary
maxlen = 200  # Maximum length of each review
batch_size = 64

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Step 2: Data Preprocessing
# Pad sequences to ensure consistent input size
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Step 3: Define the LSTM model
model = Sequential()

# Embedding layer: Converts integers into dense vectors of fixed size
model.add(Embedding(max_features, 128, input_length=maxlen))

# LSTM layer: Adds the LSTM network with 128 units
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# Dense layer: Fully connected layer with sigmoid activation for binary classification (positive/negative)
model.add(Dense(1, activation='sigmoid'))

# Step 4: Compile the Model
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(x_train, y_train, epochs=5, batch_size=batch_size, validation_data=(x_test, y_test))

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Step 7: Plot Training & Validation Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Step 8: Plot Training & Validation Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Step 9: Make Predictions (Optional)
# Sample reviews for sentiment prediction
sample_reviews = [
    "This movie was amazing! The storyline was fantastic.",
    "Worst movie I've ever seen, totally boring."
]

# Step 9.1: Preprocess the sample reviews (tokenize and pad them)
word_index = imdb.get_word_index()  # Get the word index mapping

# Function to tokenize and pad the sample reviews
def preprocess_reviews(reviews):
    # Convert reviews into tokenized integer sequences
    tokenized_reviews = [[word_index.get(word, 0) for word in review.lower().split()] for review in reviews]
    # Pad sequences to ensure consistent length
    return pad_sequences(tokenized_reviews, maxlen=maxlen)

# Preprocess the sample reviews
sample_reviews_padded = preprocess_reviews(sample_reviews)

# Step 9.2: Predict sentiment for sample reviews
predictions = model.predict(sample_reviews_padded)

# Output predictions
for review, prediction in zip(sample_reviews, predictions):
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"Review: {review}\nSentiment: {sentiment}\n")
