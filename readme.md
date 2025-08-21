## emotion-cli: Fine Tuned Emotion Classifier Command-Line Tool
emotion-cli is a command-line interface (CLI) tool for text-based emotion recognition. Built with Python, it leverages a fine-tuned DistilBERT model to classify the emotion of a given text input.
<br>
**Features**
- Emotion Recognition: Classifies text into various emotion categories (e.g., joy, sadness, anger).

- Command-Line Interface: Simple and intuitive to use directly from your terminal.

- Fine-Tuned Model: Uses a pre-trained DistilBERT model fine-tuned on an emotion dataset for high accuracy.

**Installation**

**1. Clone the repository**

`git clone https://github.com/your-username/your-repo-name.git `

`cd emotion-cli`

**2. Set up the virtual environment**

**For conda**

`conda create --name mlenv python=3.9`

`conda activate mlenv`

**Or, for venv**

`python3 -m venv venv`

`source venv/bin/activate`

**3. Install dependencies**

`pip install -e .`

### Usage

To predict the emotion of a text, run the emotion-cli command followed by your text in quotes.

`emotion-cli "I am not really happy"`

### About the Model

The core of this tool is a [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) that has been fine-tuned on a public [emotion dataset](https://huggingface.co/datasets/dair-ai/emotion). The model's weights and configuration are included in the model directory of this repository.

Note: The fine-tuning code used to train the model is not included in this repository to keep the project focused on the CLI application itself.
