import tensorflow as tf
import numpy as np
import json
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout

# Parameters (use the same as in train_model.py)
max_words = 1000
max_len = 20
embedding_dim = 16

# Load the dataset to get intents
import pandas as pd
df = pd.read_csv("data/chatbot_data.csv")
intents = df['intent'].unique()

# Create label encoder
label_encoder = LabelEncoder()
label_encoder.fit(df['intent'].values)

# Create tokenizer
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['query'].values)
word_index = tokenizer.word_index

# Create model with the same architecture
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

# Save the tokenizer
os.makedirs("model_output", exist_ok=True)
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

# Save the model (with correct extension)
model.save("model_output/intent_model.keras")

print("\nModel files saved successfully!")
print("You can now run the API server with: python api_server.py")