import tensorflow as tf
import numpy as np
import json
import pickle
import random
from pathlib import Path

class ChatbotModel:
    def __init__(self, model_dir="model_output"):
        """Initialize the chatbot model with trained model and preprocessing components."""
        model_dir = Path(model_dir)
        
        # Load model metadata
        with open(model_dir / "model_metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        # Load tokenizer
        with open(model_dir / "tokenizer.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)
        
        # Load label encoder
        with open(model_dir / "label_encoder.pickle", "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # Load trained model (with .keras extension)
        self.model = tf.keras.models.load_model(model_dir / "intent_model.keras")
        
        # Load intent responses
        with open("data/intent_responses.json", "r") as f:
            self.intent_responses = json.load(f)
        
        # Create a mapping from intent names to responses
        self.intent_to_responses = {}
        for intent_data in self.intent_responses:
            self.intent_to_responses[intent_data["intent"]] = intent_data["responses"]
        
        print(f"Loaded model with {self.metadata['num_intents']} intents")
    
    def preprocess_input(self, text):
        """Preprocess input text for model prediction."""
        # Convert to sequence
        sequences = self.tokenizer.texts_to_sequences([text])
        
        # Pad sequence
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=self.metadata["max_len"], 
            padding='post', 
            truncating='post'
        )
        
        return padded
    
    def predict_intent(self, text):
        """Predict the intent of the input text."""
        # Preprocess input
        processed_input = self.preprocess_input(text)
        
        # Get model prediction
        prediction = self.model.predict(processed_input)[0]
        
        # Get top intent
        top_intent_idx = np.argmax(prediction)
        top_intent = self.label_encoder.classes_[top_intent_idx]
        confidence = float(prediction[top_intent_idx])
        
        # Get top 3 intents with probabilities
        top_3_indices = prediction.argsort()[-3:][::-1]
        top_3_intents = [
            {
                "intent": self.label_encoder.classes_[idx],
                "probability": float(prediction[idx])
            }
            for idx in top_3_indices
        ]
        
        return {
            "top_intent": top_intent,
            "confidence": confidence,
            "top_intents": top_3_intents
        }
    
    def get_response(self, intent, confidence_threshold=0.3):
        """Get a response for the predicted intent."""
        if confidence_threshold and intent["confidence"] < confidence_threshold:
            return {
                "text": "I'm not sure I understand. Could you please rephrase that?",
                "confidence": intent["confidence"],
                "intent": intent["top_intent"],
                "generated": False
            }
        
        # Get responses for the intent
        responses = self.intent_to_responses.get(intent["top_intent"], 
            ["I'm not sure how to respond to that."])
        
        # Select a random response
        response_text = random.choice(responses)
        
        return {
            "text": response_text,
            "confidence": intent["confidence"],
            "intent": intent["top_intent"],
            "generated": True
        }
    
    def process_message(self, message):
        """Process a message and return a response."""
        # Predict intent
        intent_prediction = self.predict_intent(message)
        
        # Get response
        response = self.get_response(intent_prediction)
        
        # Add additional information for debugging
        response["all_intents"] = intent_prediction["top_intents"]
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize the model
    chatbot = ChatbotModel()
    
    # Test some queries
    test_queries = [
        "hello there",
        "can you build me a mobile app?",
        "how much do you charge for websites?",
        "I need to schedule a meeting",
        "what's your contact information",
        "thanks for your help",
        "goodbye"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = chatbot.process_message(query)
        print(f"Intent: {response['intent']} (Confidence: {response['confidence']:.4f})")
        print(f"Response: {response['text']}")
        print(f"Top intents: {response['all_intents']}")