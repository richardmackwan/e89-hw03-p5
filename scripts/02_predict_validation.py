import torch
import torch.nn.functional as F

from classifier_utils import DEVICE, MODEL_PATH, ImageClassifier, make_data_loaders


train_and_valid_data, _, valid_loader, _ = make_data_loaders()
model = ImageClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

X_batch, y_batch = next(iter(valid_loader))
X_new = X_batch[:3].to(DEVICE)
y_true = y_batch[:3]
with torch.no_grad():
    probabilities = F.softmax(model(X_new), dim=1)
    top_probabilities, top_classes = probabilities.topk(4, dim=1)

for image_index, (class_indices, class_probabilities) in enumerate(
    zip(top_classes.cpu(), top_probabilities.cpu()), start=1
):
    print(
        f"Image {image_index}: actual={train_and_valid_data.classes[y_true[image_index - 1]]}"
    )
    for class_index, probability in zip(class_indices, class_probabilities):
        class_name = train_and_valid_data.classes[class_index.item()]
        print(f"  {class_name}: {probability.item():.4f}")
