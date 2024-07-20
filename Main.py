import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from collections import Counter
import time
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


class MultimodalModel(nn.Module):
    def __init__(self, num_landmarks, num_classes):
        super(MultimodalModel, self).__init__()
        # Image processing branch using ResNet18
        self.image_branch = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = self.image_branch.classifier[1].in_features
        #self.image_branch.fc = nn.Identity()  # Remove the final classification layer
        self.image_branch.classifier[1] = nn.Linear(num_ftrs, 4)

    def forward(self, image):
        image_features = self.image_branch(image)

        return image_features


# Define a function to preprocess the image before passing it to the model
def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match the input size expected by the model
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    return transform(frame).unsqueeze(0)  # Add a batch dimension

#device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

# Load your PyTorch model
'''
models = []
for i in range(6):
    model = MultimodalModel(num_landmarks=0, num_classes=4)
    model_checkpoint = torch.load(f'model_mv{i}.pth', map_location=torch.device('cpu'))
    if isinstance(model_checkpoint, torch.nn.parallel.DataParallel):
        model_state_dict = model_checkpoint.module.state_dict()
    else:
        model_state_dict = model_checkpoint
    model.load_state_dict(model_state_dict)
    model.to(torch.device('cpu'))
    model.eval()
    models.append(model)
'''

def load_models(model_paths):
    models = []
    for path in model_paths:
        model = MultimodalModel(num_landmarks=0, num_classes=4)
        model_checkpoint = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(model_checkpoint, torch.nn.parallel.DataParallel):
            model_state_dict = model_checkpoint.module.state_dict()
        else:
            model_state_dict = model_checkpoint
        model.load_state_dict(model_state_dict)
        model.to(torch.device('cpu'))
        model.eval()
        models.append(model)
    return models

# Capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)

model_paths = ['model_mv1.pth', 'model_mv2.pth', 'model_mv3.pth', 'model_mv4.pth', 'model_mv5.pth', 'model_mv6.pth']
#model_paths = ['model_mv1.pth', 'model_mv2.pth', 'model_mv6.pth']
models = load_models(model_paths)

# Function to predict using a single model
def predict(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()

while True:
    #start_time = time.time()  # Record start time

    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess_image(frame)

    with ThreadPoolExecutor() as executor:
        # Predict using each model concurrently
        future_to_model = {executor.submit(predict, model, input_tensor): model for model in models}
        # Get predictions from each model
        y_pred_lists = [future.result() for future in future_to_model]

    # Aggregate predictions using majority voting
    print("Models Predict Probabilities", y_pred_lists)
    final_predictions = []
    for i in range(len(y_pred_lists[0])):
        votes = [preds[i] for preds in y_pred_lists]
        majority_vote = max(set(votes), key=votes.count)
        final_predictions.append(majority_vote)

    # Print the majority vote
    print("Majority Vote Engagement Level:", final_predictions[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                f"Majority Vote Engagement Level: {final_predictions[0]}",
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #elapsed_time = time.time() - start_time  # Calculate elapsed time
    #time.sleep(max(0, 0.5 - elapsed_time))  # Delay for remaining time to ensure 0.5 seconds between frames

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()