'''
Build a Multiclass classifier using the CNN model. Use MNIST or any other suitable 
dataset. a. Perform Data Pre-processing b. Define Model and perform training c. 
Evaluate Results using confusion matrix.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Step 1: Data Preprocessing

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to add the channel dimension and normalize the images
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 2: Define CNN Model

model = models.Sequential()

# Add convolutional layer 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Add convolutional layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Add convolutional layer 3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the 3D outputs to 1D for fully connected layers
model.add(layers.Flatten())

# Add fully connected (dense) layer
model.add(layers.Dense(64, activation='relu'))

# Output layer with 10 units (one for each digit) and softmax activation for multiclass classification
model.add(layers.Dense(10, activation='softmax'))

# Step 3: Compile the Model

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the Model

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Step 5: Evaluate the Model

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Step 6: Confusion Matrix

# Predict the labels for the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 7: Classification Report

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# Step 8: Visualize some test images and their predictions

fig, axes = plt.subplots(1, 5, figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Pred: {y_pred_classes[i]}\nTrue: {y_true_classes[i]}")
    ax.axis('off')
plt.show()
