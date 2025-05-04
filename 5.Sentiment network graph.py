'''
Perform Sentiment Analysis in the network graph using RNN. 
'''
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Simulated data
reviews = [
    "This product is amazing and exceeded expectations",
    "Terrible experience, will never buy again",
    "Absolutely loved the quality and the service",
    "Poor packaging and bad customer support",
    "Great value for the price",
    "Worst purchase of the year",
    "Very satisfied with the performance",
    "Disappointed with the battery life",
    "Highly recommend this to everyone",
    "Not worth the money"
]

sentiments = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
users = ["User1", "User2", "User3", "User4", "User5", "User6", "User7", "User8", "User9", "User10"]

# Step 1: Build Network Graph
G = nx.Graph()
for i in range(len(reviews)):
    user = users[i]
    review = f"Review_{i+1}"
    G.add_node(user, type='user')
    G.add_node(review, type='review', text=reviews[i], sentiment=sentiments[i])
    G.add_edge(user, review)

# Visualize the graph
plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=42)
colors = ['lightblue' if G.nodes[n].get('type') == 'user' else 'lightgreen' for n in G.nodes]
nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='gray')
plt.title("User-Review Graph")
plt.show()

# Step 2: Preprocess text
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded = pad_sequences(sequences, maxlen=20)

# Step 3: Prepare data for RNN
X = padded
y = np.array(sentiments)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define RNN model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=20),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)

# Step 6: Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.2f}")

# Step 7: Predict sentiment for each review node and update the graph
predictions = (model.predict(X) > 0.5).astype("int32").flatten()
for i in range(len(reviews)):
    review = f"Review_{i+1}"
    G.nodes[review]['predicted_sentiment'] = int(predictions[i])

# Optional: Show predicted vs actual
for i in range(len(reviews)):
    print(f"{reviews[i]}")
    print(f"Actual: {sentiments[i]}, Predicted: {predictions[i]}\n")
