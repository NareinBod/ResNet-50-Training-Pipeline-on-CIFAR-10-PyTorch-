import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as datasets 
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import random_split, DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# For viewing the images
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os 

# --- 1. DATA PREPARATION AND TRANSFORMS ---

# ResNet uses the CIFAR-10 database (50k training, 10k testing images).

# ToTensor() changes image data into a format PyTorch uses: (Channels, Height, Width).
# Normalize moves pixel values to a standard range (mean=0.5, std=0.5).

# Transforms for training data (includes extra steps for better learning)
transform_train_data = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Standardize pixel values
    transforms.RandomHorizontalFlip(),           # Randomly flip images (Data Augmentation)
    transforms.RandomCrop(32, padding=1, padding_mode="reflect") # Randomly crop images (Data Augmentation)
])

# Transforms for test data (only standard steps, no extra augmentations)
transform_test_data = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the Training Data (50,000 images)
train_data = datasets.CIFAR10 (
    root="data",
    train=True,
    download=True,
    transform=transform_train_data
)

# Load the Testing Data (10,000 images)
test_data = datasets.CIFAR10 (
    root="data",
    train=False,
    download=True,
    transform=transform_test_data
)

# --- 2. DATA SPLITTING AND DATALOADERS ---

# Percentage of data used for Validation (20%)
val_ratio = 0.2
# Randomly split the 50,000 training images into a smaller training set and a validation set
train_dataset, val_dataset = random_split(
    train_data, 
    [int((1 - val_ratio) * len(train_data)), int(val_ratio*len(train_data))]
)

# How many images the model sees at once
batch_size = 32

# Create efficient data loaders to feed data to the model in batches
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

# Print one batch's shape to confirm the format: (Batch Size, Channels, Height, Width)
for X, Y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of Y: {Y.shape} {Y.dtype}")
    break

# --- 3. DATA VISUALIZATION ---

# Function to view a batch of images
def show_batch(dataloader):
    for images, labels in dataloader:
        # Denormalize the image data so it looks correct (reverse the Normalize step)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        images = images * std + mean
        
        # Display the images
        fig, ax = plt.subplots(figsize=(10, 10))
        # make_grid puts images into a grid, permute changes the order to (H, W, C) for plotting
        ax.imshow(make_grid(images, 10).permute(1, 2, 0).numpy())
        plt.show()
        break 

# Display a sample batch of training images
# show_batch(train_dataloader)

# --- 4. DEVICE AND DATA MOVEMENT ---

# Check if a GPU (cuda) is available. If not, use the CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Helper function to move any PyTorch object (tensor, model) to the chosen device
def to_device(entity, device):
    if isinstance(entity, (list, tuple)):
        return [to_device(elem, device) for elem in entity]
    return entity.to(device, non_blocking=True)

# Wrapper class to automatically move data batches to the correct device (GPU/CPU)
class DeviceDataloader():
    def __init__(self, dataloader, device):
        self.dl = dataloader
        self.device = device

    # When you iterate (loop) over this object, it yields the data on the GPU/CPU
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
    
# Apply the wrapper to all data loaders
train__dataloader = DeviceDataloader(train_dataloader, device)
val__dataloader = DeviceDataloader(val_dataloader, device)
test__dataloader = DeviceDataloader(test_dataloader, device)

# --- 5. RESIDUAL BLOCK DEFINITION (The Core Building Block) ---

# This implements the Bottleneck Block used in ResNet-50.
# The goal is to compute a complex function while also allowing the input to skip it.
class Block(nn.Module): 
    # out_channels is the number of filters in the middle 3x3 layer.
    # identity_downsample handles changes in size/channels for the skip connection.
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4 # The final layer will have 4 times the out_channels.
        
        # 1. Bottleneck Layer (1x1 Conv): Reduces the number of channels/filters
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels) # Normalizes the output of conv1
        
        # 2. Main Feature Extraction Layer (3x3 Conv): The main layer that looks for patterns
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1) 
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 3. Expansion Layer (1x1 Conv): Increases the number of channels back up (x4)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion) # FIX: Need correct channel count, not '10'

        self.relu = nn.ReLU() # Activation function
        self.identity_downsample = identity_downsample # The skip connection transformation

    def forward(self, x):
        identity = x # Save the input for the skip connection

        # Main path processing (3 convolutional layers)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        x = self.conv3(x)
        x = self.bn3(x)

        # Apply the skip connection transformation if one is needed
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # The core idea of ResNet: Add the input (identity) to the output (x)
        x += identity

        # Apply ReLU activation to the combined result
        x = self.relu(x)

        return x

# --- 6. RESNET MODEL DEFINITION ---

# The full ResNet architecture, which uses the Block defined above.
class ResNet(nn.Module):
    # layers: a list like [3,4,6,3] that defines how many blocks are in each stage
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64 # Starting number of channels for the main path

        # --- Initial Layers ---
        # The first convolution handles the raw input image
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # MaxPool halves the image size (spatial dimension)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # --- Main Body (4 Stages) ---
        # Each layer is a stage with multiple residual blocks.
        # _make_layer handles the creation of these stages.
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)   # Output channels=256 (64*4)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)  # Halves size, Output channels=512
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)  # Halves size, Output channels=1024
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)  # Halves size, Output channels=2048
        
        # --- Final Layers ---
        # Global Average Pooling: Reduces each feature map to a single value (1x1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # Fully Connected Layer: Makes the final classification decision (10 classes)
        self.fc = nn.Linear(512 * 4, num_classes) # Input features = 2048

    # Defines the data flow through the entire network
    def forward(self, x):
        # Initial Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Main Body Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final Layers
        x = self.avgpool(x)
        # Flatten the data from (N, 2048, 1, 1) to (N, 2048)
        x = x.reshape(x.shape[0], -1) 
        x = self.fc(x)

        return x

    # Helper function to build a stage (a set of residual blocks)
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = [] 

        # --- CRITICAL DIMENSION CHECK for the FIRST block in the stage ---
        # A transformation is needed if:
        # 1. The image size is being halved (stride != 1) OR
        # 2. The channel count is changing (self.in_channels != out_channels * 4)
        if (stride != 1) or (self.in_channels != out_channels * 4):
            # Create a special skip connection layer (1x1 convolution + BatchNorm)
            # This makes the skip connection's output match the main path's size and channels.
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, stride=stride, kernel_size=1),
                nn.BatchNorm2d(out_channels*4)
            )

        # 1. Add the FIRST block (The Special Block)
        # This block uses the 'identity_downsample' to handle the size/channel change.
        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )
        
        # Update the number of input channels for the next blocks in this stage.
        self.in_channels = out_channels * 4 

        # 2. Add the REMAINING blocks (The Regular Blocks)
        # These blocks keep the size and channel count the same (stride=1, no downsampling).
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels)) 

        # Package all blocks into a single module for this stage
        return nn.Sequential(*layers)


# --- 7. MODEL INSTANTIATION AND TRAINING SETUP ---

# Create the ResNet-50 model and move it to the GPU/CPU
# layers=[3,4,6,3] defines the 50-layer structure.
model = ResNet(block=Block, layers=[3,4,6,3], image_channels=3, num_classes=10).to(device)

# Loss Function: Measures how wrong the model's predictions are.
# CrossEntropyLoss is standard for multi-class classification.
loss_fn = nn.CrossEntropyLoss()

#optimizer
# Add weight_decay to the optimizer initialization
'''weight_decay = Weight decay is essentially a method of regularization that helps prevent overfitting.

Weight Decay is a foundational concept in preventing machine learning models from becoming too reliant on the specific quirks of the training data.

'''
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


# Function to calculate the model's accuracy
def accuracy(logits, labels):
    # Find the class with the highest score (prediction)
    _, predClassId = torch.max(logits, dim=1) 
    # Compare predictions to actual labels and calculate the percentage of correct answers
    return torch.sum(predClassId == labels).item() / len(labels) # FIX: Use len(labels) for correct denominator

# Placeholder for the training loop
def train(dataloader, model, loss_fn, optimizer):
    model.train() # Set the model to training mode

    current_loss = 0.0
    current_acc = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        current_loss += (loss.item()*len(images))
        current_acc += accuracy(outputs, labels)*len(images)

    current_loss /= len(dataloader.dl.dataset)
    current_acc /= len(dataloader.dl.dataset)

    return current_loss, current_acc



def test(dataloader, model, loss_fn):
    model.eval() # Set the model to evaluation mode (turns off features like dropout)
    # Actual loop logic would go here: iterate over dataloader, compute loss/accuracy, but NO backward()
    
    total_loss = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        #loop goes here
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            total_loss += loss.item()*len(images)
            total_acc += accuracy(outputs, labels)*len(images)

    total_loss /= len(dataloader.dl.dataset)
    total_acc /= len(dataloader.dl.dataset)

    return total_loss, total_acc

#Execution

epochs = 15

#schedule: 
# A scheduler automatically changes (usually lowers) the learning rate during training.
# This helps the model learn better and can lead to higher accuracy and faster convergence.
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

for i in range(epochs):
    print(f"EPOCH: {i + 1}/{epochs}")
    train_loss, train_acc = train(train__dataloader, model, loss_fn, optimizer=optimizer)
    val_loss, val_acc = test(val__dataloader, model, loss_fn)

    # ⬇️ 3. Scheduler Step (New) ⬇️
    scheduler.step()

    print(f"Training Loss: {train_loss:>0.3f} | Training Acc: {train_acc*100:>0.2f}%")
    print(f"Validation Loss: {val_loss:>0.3f} | Validation Acc: {val_acc*100:>0.2f}%")

print("\n--- Training Complete ---")

final_test_loss, final_test_acc = test(test__dataloader, model, loss_fn=loss_fn)
print(f"Test Loss: {final_test_loss:>0.3f} | Test Accuracy: {final_test_acc*100:>0.2f}%")
    
    