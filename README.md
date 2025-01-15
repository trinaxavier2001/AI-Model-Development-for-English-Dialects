# Dialect Classification

This project focuses on building and training a machine learning model to classify regional dialects from audio files using **Wav2Vec2**, a state-of-the-art pre-trained model for speech processing. The model is trained on labeled audio data and evaluated for its accuracy in classifying dialects.

---

## Project Description

The goal of this project is to classify dialects from audio recordings using deep learning techniques. It leverages the **Wav2Vec2** model from the Hugging Face Transformers library for feature extraction and fine-tuning. This project aims to contribute to digital forensics and language recognition tasks by enabling automatic dialect classification for better insights and forensic evidence collection.

---

## Features

- **Wav2Vec2-based architecture**: Utilizes pre-trained speech models for high accuracy.
- **Support for multiple dialects**: Trains on labeled data for dialect identification.
- **Data preprocessing**: Handles audio resampling, padding, and truncation.
- **Logging**: Logs training and validation losses/accuracies for each epoch.
- **Early stopping**: Prevents overfitting during training.

---

## Prerequisites

- Python 3.8+
- GPU (optional but recommended for faster training)
- `pip` to install dependencies

---

## Installation

### Step 1: Clone the Repository
Clone the project repository from your source control platform:
```bash
git clone <repository_url>
cd <repository_name>
```
### Step 2: Set Up Virtual Environment

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

1. Ensure your dataset of audio files is organized in subfolders under the `audio_files` directory. 
   - Each subfolder should represent a class (e.g., dialect) and contain `.wav` files corresponding to that class.



2. Place the `audio_files` directory in the root folder of the project.

3. The script will automatically preprocess these files, extract features, and use them for training.

```bash
python extract_features_from_audio.py
```


### Step 5: Train the model
You can directly train the model
```bash
python modelTrain.py
```
The model will train and will be saved by the name of best_model_{epoch_number}

### Step 6: Test the model

There are 4 files in test_files. For which model will test the data
```bash
python testAudio.py
```
