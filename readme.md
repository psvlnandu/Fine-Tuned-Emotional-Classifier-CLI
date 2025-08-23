## Fine-Tuned-Emotional-Classifier-CLI
This repository showcases a professional-grade Command-Line Interface (CLI) for emotion classification. The project demonstrates a full MLOps workflow, from fine-tuning a pre-trained DistilBERT model from Hugging Face to packaging the application as a pip-installable tool. It highlights skills in Natural Language Processing (NLP), transfer learning, and model deployment
<br>
**Features**
- Emotion Recognition: Classifies text into various emotion categories (e.g., joy, sadness, anger).

- Command-Line Interface: Simple and intuitive to use directly from your terminal.

- Fine-Tuned Model: Uses a pre-trained DistilBERT model fine-tuned on an emotion dataset for high accuracy.

### Installation

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
### Option 1: Use the pre-trained model

The repository includes a pre-trained model in the trained_model directory, so you can immediately use the CLI.

`emotion-cli "I am not really happy"`
### Options 2: Fine-tune the model yourself

You can easily fine-tune the model using the provided run-all.ipynb notebook. The notebook walks you through the entire process, from training to testing.

### About the Model
1. **Open the notebook**: Open run-all.ipynb in a Jupyter environment (like Kaggle, VS Code, or JupyterLab).

2. **Run all cells**: Execute all cells in the notebook. This will:

- Install dependencies.
- Authenticate to Weights & Biases.
- Run the fine-tuning script (fine_tuning/train.py), saving the new model to trainedmodel.
- Copy the newly trained model to the model directory.
- Automatically test the CLI with your fine-tuned model.

### About this Model
The core of this tool is a [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) that has been fine-tuned on a public [emotion dataset](https://huggingface.co/datasets/dair-ai/emotion). 
The fine-tuning code and a pre-trained model are included in this repository. The run-all.ipynb notebook provides a complete and reproducible pipeline for training the model and preparing it for the CLI.

The model directory contains the latest working model for the CLI, while the fine_tuning directory holds the scripts for the training pipeline.
