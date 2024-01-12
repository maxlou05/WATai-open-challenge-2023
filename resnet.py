import torch
import torchvision
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os


label_map = {
    0: "bike",
    1: "cabinet",
    2: "chair",
    3: "coffee maker",
    4: "fan",
    5: "kettle",
    6: "lamp",
    7: "mug",
    8: "sofa",
    9: "stapler",
    10: "table",
    11: "toaster"
}


# Make predictions with the model on an unshuffled test dataset
def generate_csv(model, test_data, file_name):
    predictions = model(test_data)
    y_pred = torch.argmax(predictions, axis=-1)

    df = pd.DataFrame({
        "Index": np.arange(10800),  # This creates a list of integers from 0 to the lenght of test_data (number of pics)
        "Label": y_pred
    })
    df.to_csv(file_name, index=False)


# -------------------------------Loading data------------------------------
class SOPDataset(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train, image_transform=None):
        # Torch convolution expects data in (C, H, W) but dataset is given in (H, W, C)
        X_train = np.transpose(X_train, (0, 3, 1, 2))
        self.X_train = X_train
        self.y_train = y_train
        self.image_transform = image_transform

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, idx):
        image = self.X_train[idx, ...]
        image = torch.as_tensor(image)
        label = self.y_train[idx]
        if self.image_transform:
            image = self.image_transform(image)
        return image, label
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, X_test, image_transform=None):
        # Torch convolution expects data in (C, H, W) but dataset is given in (H, W, C)
        X_test = np.transpose(X_test, (0, 3, 1, 2))
        self.X_test = X_test
        self.image_transform = image_transform

    def __len__(self):
        return self.X_test.shape[0]

    def __getitem__(self, idx):
        image = self.X_test[idx, ...]
        image = torch.as_tensor(image)
        if self.image_transform:
            image = self.image_transform(image)
        return image


with np.load("beginner_data.npz") as data:
    X_train = data['train_images']
    y_train = data['train_labels']
    X_test = data['test_images']


train_data = SOPDataset(X_train[:52000], y_train[:52000])  # A little less than a 10% split, 52k/56k
validation_data = SOPDataset(X_train[52000:], y_train[52000:])
test_data = TestDataset(X_test)

# Divide by 255 to get data in [0, 1]
train_data.X_train = (train_data.X_train/255).astype(np.float32)
validation_data.X_train = (validation_data.X_train/255).astype(np.float32)
test_data.X_test = (test_data.X_test/255).astype(np.float32)


# -----------------------------------DATA PREPROCESSING-----------------------------------
# In pytorch, we have to create "data loaders" to group our giant list into batches ourselves
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1000)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
# No need to do normalization here? since we already do that anyways at each step with BatchNorm2D



# ------------------------------------MODEL DEFINITIONS------------------------------------
# Defining a ResNetV2 block (see https://arxiv.org/pdf/1603.05027.pdf)
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        # First, call init on super class (torch.nn.Module)
        super(ResidualBlock, self).__init__()

        # Some settings for our convolution layer (can think of as the sliding of a grid filter over our image)
        conv_kwargs = {
            "kernel_size": (3,3),  # Size of the filter
            "padding": 1,  # Padding to add on the outside of our image (encircle it with one more), since it takes in a 3x3 and outputs a 1x1 at the middle coordinate
            "bias": False
        }

        # Init our variables
        self.stride = stride  # How much we move our little grid filter thing each step
        self.in_channels = in_channels  # This model requires a copy of the original input data/vector to add on to the results at the end after processing
        self.channels = channels  # This is the data the we are going to process and change throughout all the math

        # Initialize all the layer types we are going to use (note, all our stuff is 2d since our data is a 2d image, we don't care how many channels each pixel has)
        # Instead, we actually define the amount of channels per pixel we output in the given channels parameter
        self.bn1 = torch.nn.BatchNorm2d(in_channels)  # This is the first required layer of ResNetV2
        self.relu = torch.nn.ReLU()  # Just a relu
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=channels, stride=stride, **conv_kwargs)  # in_channels(original input) -> channels(output). Also, recall that **dict means dereference the dict, put in all the things as kwargs
        self.bn2 = torch.nn.BatchNorm2d(channels)  # The second normalization, but this time we use the output from before, since we want to keep the original input
        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, **conv_kwargs)  # channels(previous output) -> channels(output). I guess this time we don't need to change the stride anymore, otherwise we are double shrinking

    # This basically just changes the input's size to match the output's size by filling in the non-needed stuff with 0, so that you can add it later to the output
    # This is due to when stride != 1, then the output of the conv2d is going to be input/stride, as it takes in all the pixels it sees and outputs only 1 value
    def strided_identity(self, x):
        # This tells it how to apply the stride???
        # Downsample with the 'nearest' method  (striding if dims are divisible by stride)
        if self.stride != 1:
            # Downsampling is a signal processing technique to reduce sample rate (I suppose the amount of data to be processed in this case)
            # It discards some and transforms other data points (see downsampling, involves anti-ailiasing and decimation (ie taking every nth data sample))
            # This makes it the correct image size (ie correct amount of pixels)
            x = torch.nn.functional.interpolate(x, mode="nearest", scale_factor=(1/self.stride))  # However, not sure what this scale factor is, it probably determines n = self.stride since number of output data = input * 1/self.stride
        
        # Create padding tensor for extra channels
        # This makes it the correct channel depth (ie correct amount of channels per pixel (we might evolve from 3channel rgb to more things or less things))
        if self.channels != self.in_channels:
            (b, c, h, w) = x.shape  # batch, col, height, width?
            num_pad_channels = self.channels - self.in_channels  # Determine the number of channels
            pad = torch.zeros((b, num_pad_channels, h, w), device=x.device)  # make them all zeros
            x = torch.cat((x, pad), dim=1)  # Add the paddings to the downsampled stuff (I guess net effect is just replaced a whole bunch of data with 0s)
        return x
    
    # IMPORTANT: this is the function that defines what happens in what order when you use this block/model
    def forward(self, x):
        # Compute "residual" (this is to be added to the output at the end)
        identity = self.strided_identity(x)
        
        # Time to actually run the ResNetV2 block, but need to save the results to a separate variable since we need to keep the original input x
        z = self.bn1(x)
        z = self.relu(z)
        z = self.conv1(z)
        
        z = self.bn2(z)
        z = self.relu(z)
        z = self.conv2(z)

        # Now add the residual to the data that went through the 'training' or convolution
        out = identity + z
        return out



# Defining the entire CNN (Convolutional Neural Network) architecture (which uses ResNetV2 blocks in it)
class ResNetV2(torch.nn.Module):
    # Default we have rgb 3 channels, and our input image size is 64x64
    def __init__(self, in_channels=3, in_shape=(64, 64)):
        # Guess you always need to do this to make sure the module has all the stuff it needs
        super().__init__()
        # Init our variables
        self.in_channels = in_channels
        self.in_shape = in_shape

        # Initalize all the layer types, as usual
        self.input_conv = torch.nn.Conv2d(3, 32, kernel_size=(7, 7), bias=False, padding=3)  # We first start the thing with a conv2d before using all the resnet blocks
        self.input_bn = torch.nn.BatchNorm2d(32)  # Then we gotta normalize, and make sure that the input number of features per pixel is the same as the output number of features from the first conv2d
        # Next, the real meat of the model, where we will use our residual blocks defined earlier
        self.layer1 = ResidualBlock(32, 32)  # Shape (B, 32, 64, 64)
        self.layer2 = ResidualBlock(32, 32)  # Shape (B, 32, 64, 64)
        self.layer3 = ResidualBlock(32, 64, stride=2)  # Shape (B, 64, 32, 32)
        self.layer4 = ResidualBlock(64, 64)  # Shape (B, 64, 32, 32)
        self.layer5 = ResidualBlock(64, 128, stride=2)  # Shape (B, 128, 16, 16)
        self.layer6 = ResidualBlock(128, 128)  # Shape (B, 128, 16, 16)
        self.layer7 = ResidualBlock(128, 256, stride=2) # Shape (B, 256, 8, 8)
        self.layer8 = ResidualBlock(256, 256) # Shape (B, 256, 8, 8)
        self.layer9 = ResidualBlock(256, 512, stride=2)
        self.layer10 = ResidualBlock(512, 512)
        # Now, for the output layers, how to decide which one we are predicting
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))  # We average pool? the pictures (8x8 now) into just a 1x1 thing (our image has now only 1 pixel left, but 128 features)
        self.output_layer1 = torch.nn.Linear(512, 256)
        self.output_layer2 = torch.nn.Linear(256, 64)  # Finally, we do some transformations on the features to decide what picture it is
        self.output_layer3 = torch.nn.Linear(64, 12)

    def forward(self, x):
        '''
        param x: Tensor with shape (B, 3, 64, 64)
        returns: Tensor with shape (B, 12)
        '''
        # Run the input layers
        x = self.input_conv(x)
        x = self.input_bn(x)
        # Run the main training stuffs
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # x = self.layer9(x)
        # x = self.layer10(x)
        # Output layers
        x = self.pool(x)
        x = x.squeeze()
        # x = self.output_layer1(x)
        x = self.output_layer2(x)
        x = self.output_layer3(x)
        return x



# --------------------------------------HYPERPARAMETERS-------------------------------------
LEARNING_RATE = 1e-3
flag = False  # We want to decrease the learning rate after we are close to the value?
flag2 = False
LEARNING_RATE2 = 1e-4
LEARNING_RATE3 = 2e-5
NUM_EPOCHS = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

model = ResNetV2().to(DEVICE)  # Send the model to be computed on the GPU rather than CPU
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()  # The satandard loss function for multi classification



# ---------------------------------------TRAIN MODEL---------------------------------------
# Setting up the training loop. I suppose if epoch = -1, it never stops?
def train(model, train_loader, loss_fn, optimizer, device='cpu', epoch=-1):
    """
    Trains a model for one epoch (one pass through the entire training data).

    :param model: PyTorch model
    :param train_loader: PyTorch Dataloader for training data
    :param loss_fn: PyTorch loss function
    :param optimizer: PyTorch optimizer, initialized with model parameters
    :kwarg epoch: Integer epoch to use when printing loss and accuracy
    :returns: Accuracy score
    """
    total_loss = 0
    all_predictions = []
    all_targets = []

    model = model.to(device)  # Do this again I guess?
    model.train()  # Set the model into training mode
    # Loop through the data, which the trian loader helps us put into batches
    for i, (inputs, targets) in enumerate(train_loader):  # 1. Fetches the next batch of data
        inputs = inputs.to(device)  # Send our inputs to the GPU
        targets = targets.to(device)  # I guess send labels to the GPU?

        optimizer.zero_grad()  # 2. Reset the all the gradients to zero
        outputs = model(inputs)  # 3. Compute the model's outputs
        loss = loss_fn(outputs, targets)  # 4. Compute the loss (ie how wrong we are)
        loss.backward()  # 5. Backpropagtion to adjust weights and biases?
        optimizer.step()  # 6. Apply gradient descent and actually adjust the weights and biases for real?

        # Track some values
        total_loss += loss.item()
        preds = torch.argmax(outputs.cpu(), dim=-1)  # Take the class with the highest probability/output as the prediction. Also bring outputs back to the CPU first
        all_predictions.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        # Print the stats
        if i % 100 == 0:
            running_loss = total_loss/(i+1)
            print(f"Epoch {epoch + 1}, batch {i + 1}: loss = {running_loss:.2f}")

    acc = accuracy_score(all_targets, all_predictions)  # As in how many we got right

    # Print average loss and accuracy
    print(f"Epoch {epoch + 1} done. Average train loss = {total_loss / len(train_loader):.2f}, average train accuracy = {acc * 100:.3f}%")
    return acc


# Validate - evaluate the validation data
def validate(model, validation_loader, device='cpu', epoch=-1):
    """
    Implements validation accuracy checking, runs through 1 epoch
    """
    all_predictions = []
    all_targets = []

    model = model.to(device)
    model.eval()
    for i, (inputs, targets) in enumerate(validation_loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs.cpu(), dim=-1)
            all_predictions.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    acc = accuracy_score(all_targets, all_predictions)  # Validation accuracy
    return acc


# Testing the model - outputting the predictions
def test(model, test_loader, device='cpu'):
    """
    Tests a model for one epoch of test data.
    This one does not evalulate how well we did on the testing set, since truth labels are not given

    :param model: PyTorch model
    :param test_loader: PyTorch Dataloader for test data

    :returns: List of predictions
    """
    all_predictions = []
    model = model.to(device)
    model.eval()  # Set model in evaluation mode rather than training mode
    for i, inputs in enumerate(test_loader):  # 1. Fetch next batch of data
        inputs = inputs.to(device)
        with torch.no_grad():  # Since we are not training, don't need to do any adjusting of weights
            outputs = model(inputs)  # Since we are in evaluate mode, this doens't train it, just evaluates it
            preds = torch.argmax(outputs.cpu(), dim=-1).tolist()
            all_predictions.extend(preds)
    return all_predictions


# --------------------------------------GET RESULTS----------------------------------------
train_metrics = []
val_metrics = []
best_accs = [0.0, 0.0, 0.0]
PATIENCE = 5
BEST_MODEL = "resnet_best_model.pth"
patience = 0

for epoch in range(NUM_EPOCHS):
    train_acc = train(model, train_loader, loss_fn, optimizer, device=DEVICE, epoch=epoch)
    train_metrics.append(train_acc)
    val_acc = validate(model, validation_loader, device=DEVICE, epoch=epoch)
    val_metrics.append(val_acc)
    print(f"validation accuracy: {val_acc*100:.3f}%")
    
    # Implement a simple early stopping
    if(val_acc > best_accs[0]):
        best_accs.pop()
        best_accs.insert(0, val_acc)
        patience = 0  # We want [PATIENCE] decreases in a row to stop
        os.remove("resnet_model_2.pth")
        os.rename("resnet_model_1.pth", "resnet_model_2.pth")
        os.rename("resnet_model_0.pth", "resnet_model_1.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict()
        }, f"resnet_model_0.pth")
    elif(val_acc > best_accs[1]):
        best_accs.pop()
        best_accs.insert(1, val_acc)
        patience = 0  # We want [PATIENCE] decreases in a row to stop
        os.remove("resnet_model_2.pth")
        os.rename("resnet_model_1.pth", "resnet_model_2.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict()
        }, f"resnet_model_1.pth")
    elif(val_acc > best_accs[2]):
        best_accs.pop()
        best_accs.insert(2, val_acc)
        patience = 0  # We want [PATIENCE] decreases in a row to stop
        os.remove("resnet_model_2.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict()
        }, f"resnet_model_2.pth")
    else:
        # The first time this happens means we are probably close to the optimum
        if not flag:
            print("Switching learning rate!")
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE2)
            flag = True
        elif not flag2:
            print("Switching learning rate again!")
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE3)
            flag2 = True
        patience += 1
    # if(patience == PATIENCE):
    #     break


print("training accuracies:", train_metrics)
print("validation accuracies:", val_metrics)

# Save last output
predictions = test(model, test_loader, device=DEVICE)
df = pd.DataFrame({
        "Index": np.arange(10800),  # This creates a list of integers from 0 to the lenght of test_data (number of pics)
        "Label": predictions
    })
df.to_csv("resnet_k5_b128_final.csv", index=False)


# Save best outputs
for i in range(len(best_accs)):
    checkpoint = torch.load(f"resnet_model_{i}.pth")
    best_epoch = checkpoint['epoch']
    out_model = ResNetV2()
    out_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Performance {i} was in epoch", best_epoch)

    predictions_best = test(out_model, test_loader, device=DEVICE)
    df = pd.DataFrame({
            "Index": np.arange(10800),  # This creates a list of integers from 0 to the lenght of test_data (number of pics)
            "Label": predictions_best
        })
    df.to_csv(f"resnet_model_{i}.csv", index=False)
