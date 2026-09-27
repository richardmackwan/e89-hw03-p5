import json

import matplotlib.pyplot as plt

from classifier_utils import HISTORY_PATH


with HISTORY_PATH.open(encoding="utf-8") as history_file:
    history = json.load(history_file)

epochs = len(history["train_metrics"])
plt.plot(
    [epoch + 0.5 for epoch in range(epochs)],
    history["train_metrics"],
    ".--",
    label="Training",
)
plt.plot(
    [epoch + 1.0 for epoch in range(epochs)],
    history["valid_metrics"],
    ".-",
    label="Validation",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Fashion MNIST: Training and Validation Accuracy")
plt.xticks(range(1, epochs + 1))
plt.ylim(0.0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy.png", dpi=150)
plt.close()
print("Saved training_accuracy.png")
