import json

import torch
import torch.nn as nn
import torchmetrics

from classifier_utils import (
    ARTIFACTS_DIR,
    DEVICE,
    HISTORY_PATH,
    MODEL_PATH,
    N_CLASSES,
    N_EPOCHS,
    ImageClassifier,
    make_data_loaders,
    train2,
)


_, train_loader, valid_loader, _ = make_data_loaders()
torch.manual_seed(42)
model = ImageClassifier().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=N_CLASSES).to(DEVICE)

history = train2(
    model, optimizer, criterion, accuracy, train_loader, valid_loader, N_EPOCHS
)

ARTIFACTS_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
with HISTORY_PATH.open("w", encoding="utf-8") as history_file:
    json.dump(history, history_file, indent=2)

print(f"Device: {DEVICE}")
print(f"Saved model weights to {MODEL_PATH}")
print(f"Saved training history to {HISTORY_PATH}")
