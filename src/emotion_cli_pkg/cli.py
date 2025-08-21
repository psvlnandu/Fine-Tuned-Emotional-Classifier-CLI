#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os


current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
MODEL_PATH = os.path.join(project_root_dir, 'model')

normalized_path = os.path.normpath(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(normalized_path)
tokenizer = AutoTokenizer.from_pretrained(normalized_path)

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
def main():
    parser = argparse.ArgumentParser(description="Emotion Prediction CLI Tool")
    parser.add_argument("text", type=str, help="The text to analyze.")
    args = parser.parse_args()
    
    emotion = predict_emotion(args.text)
    print(f"The predicted emotion is: {emotion}")

# The __name__ == "__main__" block is now just a single line
if __name__ == "__main__":
    main()