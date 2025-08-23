#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

#change this path to your opdir model
# MODEL_PATH = '/content/drive/My Drive/psvlDocuments/CLIFiles/'
# Change the path to your local model folder
MODEL_PATH = './'

# Load the model and tokenizer outside the main function
# This is more efficient for repeated use
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def predict_emotion(text):
    """Predicts the emotion of a given text."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get the predicted label
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    predicted_class_id = predictions.argmax().item()
    return labels[predicted_class_id]

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Emotion Prediction CLI Tool")
    parser.add_argument("text", type=str, help="The text to analyze.")
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Get the emotion prediction and print it
    emotion = predict_emotion(args.text)
    print(f"The predicted emotion is: {emotion}")