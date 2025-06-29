import os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../model")
model_bert = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)

# Load ResNet
model_image = models.resnet18 (weights=None)
model_image.fc = nn.Linear(model_image.fc.in_features, 2)
model_image.load_state_dict(torch.load(f"{LOCAL_MODEL_DIR}/resnet18_blood_classify.pt", map_location=device))
model_image.to(device)
model_image.eval()
