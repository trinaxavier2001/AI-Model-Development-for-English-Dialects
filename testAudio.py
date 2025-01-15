from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import torchaudio

import os

# Constants
MODEL_PATH = "saved_model/best_model_epoch_8.pth"  # Update with your saved model path
LABEL_MAPPING = {
    0: "midlands_female", 1: "scottish_male", 2: "midlands_male", 3: "welsh_male",
    4: "southern_female", 5: "irish_male", 6: "southern_male", 7: "welsh_female",
    8: "northern_male", 9: "scottish_female", 10: "northern_female"
}  # Update with your label mapping

# Load model and processor
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=len(LABEL_MAPPING))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_audio(audio_path, target_sample_rate=16000, max_audio_length=16000*10):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)  # Convert to mono if stereo

        # Pad or truncate
        if waveform.shape[0] > max_audio_length:
            waveform = waveform[:max_audio_length]
        elif waveform.shape[0] < max_audio_length:
            waveform = torch.nn.functional.pad(waveform, (0, max_audio_length - waveform.shape[0]), mode="constant", value=0)

        return waveform
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None
def predict_dialect(audio_path):
    waveform = preprocess_audio(audio_path)
    if waveform is None:
        return "Error processing audio file."

    # Process the waveform with the Wav2Vec2 processor
    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Forward pass through the model
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label_idx = torch.argmax(logits, dim=1).item()
    predicted_label = LABEL_MAPPING[predicted_label_idx]

    return predicted_label


# Define the directory containing test audio files
TEST_FILES_DIR = "test_files"  # Replace with your actual directory path

# Get all .wav files in the test directory
test_audio_files = [os.path.join(TEST_FILES_DIR, f) for f in os.listdir(TEST_FILES_DIR) if f.endswith('.wav')]

# Loop through each audio file and predict the dialect
for audio_file in test_audio_files:
    result = predict_dialect(audio_file)
    print(f"File: {audio_file} -> Predicted Dialect: {result}")
