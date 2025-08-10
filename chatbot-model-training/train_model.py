import numpy as np
import pandas as pd
import tensorflow as tf
import json
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
df = pd.read_csv("data/chatbot_data.csv")
print(f"Loaded {len(df)} training examples")

# Prepare text data
X = df['query'].values
y = df['intent'].values

# Convert intents to numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create and configure tokenizer
max_words = 1000  # Maximum vocabulary size
max_len = 20      # Maximum sequence length

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
print(f"Vocabulary size: {len(word_index)}")

# Convert text to sequences
X_sequences = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform length
X_padded = pad_sequences(X_sequences, maxlen=max_len, padding='post', truncating='post')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Create the model
embedding_dim = 16
model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model_output/training_history.png')
plt.close()

# Analyze model predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('model_output/confusion_matrix.png')
plt.close()

# Save classification report
report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('model_output/classification_report.csv')
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Save the model and preprocessing components
model.save("model_output/intent_model")

# Save the tokenizer
with open("model_output/tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

# Save the label encoder
with open("model_output/label_encoder.pickle", "wb") as f:
    pickle.dump(label_encoder, f)

# Save model metadata
model_metadata = {
    "max_words": max_words,
    "max_len": max_len,
    "embedding_dim": embedding_dim,
    "num_intents": len(label_encoder.classes_),
    "intents": label_encoder.classes_.tolist()
}

with open("model_output/model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

print("\nModel and preprocessing components saved successfully!")