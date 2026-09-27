# Assignment 03, Problem 5

Implemented a Fashion MNIST image classifier based on the reference notebook's "Building an Image Classifier with PyTorch" section.

- Uses `torchvision.transforms.v2`, a seeded 55,000/5,000 training/validation split, batch size 32, and CUDA → MPS → CPU device selection.
- Defines the 784 → 300 → 100 → 10 ReLU MLP and trains it for 20 epochs with cross-entropy loss, SGD at learning rate 0.1, and `torchmetrics.Accuracy` through the `train2` and `evaluate_tm` helpers.
- Predicts on three validation images, reports each image's top four softmax probabilities, and counts trainable parameters.
- Plots per-epoch training and validation accuracy to `training_accuracy.png`.
- Provides ordered runnable scripts in `scripts/` and an independent end-to-end notebook in `e89_Kwan_Richard_HW03_Prob5.ipynb`.

## Run Order

Install dependencies from `requirements.txt`, then run these commands from the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/01_train_classifier.py
python scripts/02_predict_validation.py
python scripts/03_count_parameters.py
python scripts/04_plot_training_accuracy.py
```

The first run downloads Fashion MNIST into `datasets/`. Model weights and epoch history are written under `artifacts/`; these generated files are excluded from version control. The plot is written to the repository root as `training_accuracy.png`.
