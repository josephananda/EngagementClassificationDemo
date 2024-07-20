import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from collections import Counter

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
model = MultimodalModel(num_landmarks=0, num_classes=4)  # Instantiate the MultimodalModel class
model_checkpoint = torch.load('model_mv1.pth', map_location=torch.device('cpu'))  # Load model weights

if isinstance(model_checkpoint, torch.nn.parallel.DataParallel):
    model_state_dict = model_checkpoint.module.state_dict()
else:
    model_state_dict = model_checkpoint

model.load_state_dict(model_state_dict)
model.to(torch.device('cpu'))
#model.eval()  # Set the model to evaluation mode
#model = torch.load('model_mv1.pth')  # Load your trained model
model.eval()  # Set the model to evaluation mode


for param_name, param in model.named_parameters():
    if param.device.type == 'cuda':
        param.data = param.data.cpu()
        if param.grad is not None:
            param.grad.data = param.grad.data.cpu()

for buf_name, buf in model.named_buffers():
    if buf.device.type == 'cuda':
        buf.data = buf.data.cpu()


# Capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess_image(frame)

    # Pass the preprocessed frame through the model
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class
    _, predicted = torch.max(output, 1)

    # Print the predicted class (assuming the model outputs class indices)
    print("Predicted Engagement Level:", predicted.item())
    print("Predicted Engagement Prob:", output)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()