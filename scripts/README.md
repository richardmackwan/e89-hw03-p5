# Script Order

Run these from the repository root after installing `../requirements.txt` (or `requirements.txt` from the root):

1. `01_train_classifier.py` downloads Fashion MNIST as needed, trains the model for 20 epochs, and saves weights plus metric history in `artifacts/`.
2. `02_predict_validation.py` loads the trained weights and prints the top-four softmax probabilities for three validation examples.
3. `03_count_parameters.py` reports the classifier's trainable parameter count.
4. `04_plot_training_accuracy.py` plots both accuracy series and saves `training_accuracy.png` at the repository root.

`classifier_utils.py` holds the shared data setup, classifier definition, device choice, and training/evaluation helpers used by the scripts. The notebook includes its own implementation and does not import this module.
