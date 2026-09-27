from classifier_utils import ImageClassifier


model = ImageClassifier()
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(f"Trainable parameters: {parameter_count:,}")
