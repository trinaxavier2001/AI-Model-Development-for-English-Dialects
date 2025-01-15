import os
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Constants
AUDIO_DIR = "audio_files"
BATCH_SIZE = 64
MAX_DATA_POINTS = 700
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10 * TARGET_SAMPLE_RATE
MODEL_SAVE_DIR = "saved_model"
LOG_DIR = "logs"

# Create log directory if it doesn't exist
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, "training_logs.txt")

# Initialize Wav2Vec2 Processor
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# Logging lists
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Helper function to pad or truncate audio
def pad_or_truncate(waveform, target_length=MAX_AUDIO_LENGTH):
    if waveform.shape[0] > target_length:
        return waveform[:target_length]
    elif waveform.shape[0] < target_length:
        padding = target_length - waveform.shape[0]
        return torch.nn.functional.pad(waveform, (0, padding), mode="constant", value=0)
    return waveform

# Dataset Class
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, target_sample_rate=TARGET_SAMPLE_RATE, max_audio_length=MAX_AUDIO_LENGTH):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.max_audio_length = max_audio_length
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=target_sample_rate)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self.target_sample_rate:
                waveform = self.resampler(waveform)
            waveform = waveform.mean(dim=0)
            waveform = pad_or_truncate(waveform, target_length=self.max_audio_length)
            return waveform, label
        except Exception as e:
            print(f"Error loading file {audio_path}: {e}")
            return None, None

# Helper function for dynamic padding
def collate_fn(batch):
    inputs, labels = zip(*[(item[0], item[1]) for item in batch if item[0] is not None])
    inputs = torch.stack(inputs)
    inputs = processor(inputs.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True).input_values
    return inputs, torch.tensor(labels, dtype=torch.long)

# Helper function to load file paths and labels
def load_audio_files(base_dir, max_data_points):
    categories = os.listdir(base_dir)
    file_paths = []
    labels = []
    label_mapping = {}
    for idx, category in enumerate(categories):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue
        files = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.endswith(".wav")]
        random.shuffle(files)
        files = files[:max_data_points]
        file_paths.extend(files)
        labels.extend([idx] * len(files))
        label_mapping[idx] = category
    return file_paths, labels, label_mapping

# Load file paths and labels
file_paths, labels, label_mapping = load_audio_files(AUDIO_DIR, MAX_DATA_POINTS)
print(f"Loaded {len(file_paths)} audio files across {len(label_mapping)} categories.")

# Train-Test Split
train_paths, val_paths, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=SEED, stratify=labels)

# Create Datasets and Dataloaders
train_dataset = AudioDataset(train_paths, train_labels)
val_dataset = AudioDataset(val_paths, val_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Model Definition
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_mapping))
model.to(DEVICE)

# Loss Function and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Early Stopping Variables
best_val_accuracy = 0
patience = 5
patience_counter = 0

# Training Function
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    global best_val_accuracy, patience_counter
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), accuracy=correct / total)

        # Calculate and log training metrics
        train_accuracy = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")

        # Validation step
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save the best model and implement early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, f"best_model_epoch_{epoch+1}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save logs to file
        with open(LOG_FILE, "a") as log_file:
            log_file.write(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
                f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}\n"
            )

# Validation Function
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    return val_loss, val_accuracy

# Function to save the model
def save_model(model, filename):
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_path = os.path.join(MODEL_SAVE_DIR, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

# Train and Validate
train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS)
