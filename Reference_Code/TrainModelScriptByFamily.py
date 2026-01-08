## This script is designed to train a dynamic model 
#which classifies RF based what family of modulation the IQ sample belongs to


####TODO#####
##-save plots of training progress

from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.transforms.transforms import ComplexTo2D
import os
import numpy as np
from torch import Tensor
import argparse
print("Initial Imports Complete")


parser = argparse.ArgumentParser(description='Your program description here.')


#print(TorchSigSignalLists.fsk_signals)
#print(TorchSigSignalLists.ofdm_signals)
#print(TorchSigSignalLists.constellation_signals)
#print(TorchSigSignalLists.fm_signals)
#print(TorchSigSignalLists.lfm_signals)
#print(TorchSigSignalLists.chirpss_signals)
#print(TorchSigSignalLists.am_signals)

parser.add_argument('--family', default = "am qam chirp ask psk fsk fm", type = str, help='Input the target modulation families to classify on. Options are fsk, ofdm, constellation, fm, lfm, chirpss, am')
parser.add_argument('--to_delete', action="store_true", help="if included deletes dataset after training" )
parser.add_argument('--fft_size', default = 128, type = int, help="base size of FFT, used to calculate input size")
parser.add_argument('--num_samples', default = 300, type = int, help="number of samples per class of modulation")
parser.add_argument('--output_channels', default = 16, type = int, help = "the number of output channels used as basis for neural layers")
parser.add_argument('--batch_size', default=16, type = int, help="batch size for model training")
parser.add_argument('--epochs', default=30, type = int, help="number of epochs to train for")
parser.add_argument('--lr', default=0.0001, type = float, help ="learning rate for model training" )
parser.add_argument('--train_by_family', action="store_true", help="if included the model groups modulation by family targeted based on input to --family imput. Otherwise by class based on --classes input")
parser.add_argument('--total_classes', default="all", type=str, help="the list of classes of specific modulations to include in the dataset")

args = parser.parse_args()
#print(args.input)  # Access input argument

del_dataset_when_complete = args.to_delete # a boolean to determine whether to delete the generated data after model training
fft_size = args.fft_size
num_iq_samples = fft_size ** 2
num_iq_samples_dataset = num_iq_samples
num_samples_per_class = args.num_samples # the number of samples for each modulation in the class list to create
percent_validation = 0.2 # the number of validation data points to generate as a percentage of the num of train data points
class_list = TorchSigSignalLists.all_signals ##!!! Going to use all modulation types to add additional noise to data
output_channels = args.output_channels
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr #learning rate
balance = True #varaibel for tracking whther to re-balance dataset

print("Modulations Recieved: ", args.family)

print("*"*30)
print("Beginning Data Generation")
target_type = "family"
if target_type == "family":
    tf = args.family.split(' ')

target_family = tf
print("target family after split: ", tf)
print("target family after split: ", target_family)

print("target family after split len: ", len(tf))
print("target family after split len: ", len(target_family))

if len(target_family)>=3:
    balance = False

target_family_string = "Target(s)"
for x in target_family:
    target_family_string = target_family_string+"_"+str(x)
root = f"./datasets/ModulationByFamily/fft_size_{fft_size}-num_iq_samples_dataset_{num_iq_samples_dataset}"+target_family_string
os.makedirs(root, exist_ok=True)
os.makedirs(root + "/train", exist_ok=True)
os.makedirs(root + "/val", exist_ok=True)
#os.makedirs(root + "/test", exist_ok=True)


conv_config = [
    (output_channels, 2, 2, 1, True),   # 1st Conv Layer -> MaxPool
    (output_channels*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2, 2, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2, 2, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2, 2, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2, 2, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 2, 1, 1, False),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 2, 1, 1, False),   # 2nd Conv Layer -> MaxPool
    
]

# Format: list of output features for each fully connected layer
#fc_config = [output_channels*2*2*2*2, output_channels] ##!!!NOTE: this is good for smaller models
fc_config = [output_channels*2*2*2*2*2, output_channels*2*2, output_channels]



print(f"Creating dataset with fft_size {fft_size}")
print(f" IQ samples per datapoint {num_iq_samples_dataset}")
print(f" Samples per modulation class {num_samples_per_class}")
print(f" Percent used for validation {percent_validation}")
print(f" Target modulations {target_family}")

print("*"*30)
family_list = []
family_list.append(TorchSigSignalLists.fsk_signals)
family_list.append(TorchSigSignalLists.ofdm_signals)
family_list.append(TorchSigSignalLists.constellation_signals)
family_list.append(TorchSigSignalLists.fm_signals)
family_list.append(TorchSigSignalLists.lfm_signals)
family_list.append(TorchSigSignalLists.chirpss_signals)
family_list.append(TorchSigSignalLists.am_signals)
print(f"Family list : {family_list}")
print(f"Family list length : {len(family_list)}")

#num_classes = len(class_list)
##!!! Setting num classes to 2 because only looking for one modulation
num_classes = len(target_family)+1

num_samples_train = len(class_list) * num_samples_per_class # 
num_samples_val = int(len(class_list) * num_samples_per_class * percent_validation)
impairment_level = 0
seed = 123456789
 # IQ-based mod-rec only operates on 1 signal
num_signals_max = 1
num_signals_min = 1

# ComplexTo2D turns a IQ array of complex values into a 2D array, with one channel for the real component, while the other is for the imaginary component
transforms = [ComplexTo2D()]

class_name_list = target_family.copy() #create a variable to contain the positional encoded class options
class_name_list.append("other/unknown")

print(f" Number of classes {num_classes}")
print(f" Name of each class {class_name_list}")

print("*"*30)
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import DatasetCreator

dataset_metadata = DatasetMetadata(
    num_iq_samples_dataset = num_iq_samples_dataset,
    fft_size = fft_size,
    class_list = class_list,
    num_signals_max = num_signals_max,
    num_signals_min = num_signals_min,
)

train_dataset = TorchSigIterableDataset(dataset_metadata, transforms=transforms, target_labels=None)
val_dataset = TorchSigIterableDataset(dataset_metadata, transforms=transforms, target_labels=None)

train_dataloader = WorkerSeedingDataLoader(train_dataset, batch_size=1, collate_fn = lambda x: x)
val_dataloader = WorkerSeedingDataLoader(val_dataset, collate_fn = lambda x: x)

#print(f"Data shape: {data.shape}")
#print(f"Targets: {targets}")
# next(train_dataset)

dc = DatasetCreator(
    dataloader=train_dataloader,
    root = f"{root}/train",
    overwrite=True,
    dataset_length=num_samples_train
)
dc.create()


dc = DatasetCreator(
    dataloader=val_dataloader,
    root = f"{root}/val",
    overwrite=True,
    dataset_length=num_samples_val
)
dc.create()

train_dataset = StaticTorchSigDataset(
    root = f"{root}/train",
    target_labels=["class_index"]
)
val_dataset = StaticTorchSigDataset(
    root = f"{root}/val",
    target_labels=["class_index"]
)
print("Dataset Created")
print("*"*30)


def get_mod_class(label, class_list, family_dict):
    return family_dict[class_list[label]]

from collections import Counter
##Trying nesting the arrays
train_data = []
train_label = []
for x in train_dataset:
    #hold = (x[0],x[1])
    train_data.append([x[0]])
    train_label.append(x[1])

##set all labels not target modulation to 1
#new_train_label = [0 if x == 0 else 1 for x in train_label]
new_train_label = [target_family.index(get_mod_class(x, class_list,TorchSigSignalLists.family_dict)) if get_mod_class(x, class_list,TorchSigSignalLists.family_dict) in target_family else num_classes-1 for x in train_label]


val_data = []
val_label = []
for x in val_dataset:
    #hold = (x[0],x[1])
    val_data.append([x[0]])
    val_label.append(x[1])



##set all labels not target modulation to 1
new_val_label = [target_family.index(get_mod_class(x, class_list,TorchSigSignalLists.family_dict)) if get_mod_class(x, class_list,TorchSigSignalLists.family_dict) in target_family else num_classes-1 for x in val_label]

print("Dataset Formatted")
print("*"*30)
print(" Statistics of dataset classes pre balancing: ")
# Create frequency dictionary
train_freq = Counter(new_train_label)
val_freq = Counter(new_val_label)
print(f" Train data statisitics: {train_freq}")
print(f" Val data statisitics: {val_freq}")


print("*"*30)
if balance:
    print("Balancing Train Data")

    print(len(train_data))
    #identify all the target data
    print("target_family: ",target_family)
    print("len(target_family): ",len(target_family))
    condition = np.array(new_train_label) < len(target_family)
    target_new_train_label = np.array(new_train_label)[condition]
    target_train_data = np.array(train_data)[condition]

    #save the number of target data point to use to balance the non-target data
    num_balanced_samples = len(target_train_data) 
    print("Num of Target Samples: ",num_balanced_samples)
    condition = np.array(new_train_label) == len(target_family)
    excess_new_train_label = np.array(new_train_label)[condition]
    excess_train_data = np.array(train_data)[condition]

    print("len(excess_new_train_label): ",len(excess_new_train_label))

    random_choice_indices = np.random.choice(
            len(excess_new_train_label),
            size=num_balanced_samples, 
            replace=False
        )

    print("Length random_choice_indices: ", len(random_choice_indices))

    excess_new_train_label = excess_new_train_label[random_choice_indices]
    excess_train_data = excess_train_data[random_choice_indices]

    train_data = np.concatenate((target_train_data,excess_train_data), axis = 0)
    new_train_label = np.concatenate((target_new_train_label,excess_new_train_label), axis = 0)


    print("length balanced train data: ", len(train_data))
    print("length balanced train data labels: ", len(new_train_label))


    print("*"*30)
    print("Balancing Validation Data")

    print(len(val_data))
    #identify all the target data
    condition = np.array(new_val_label) < len(target_family)
    target_new_val_label = np.array(new_val_label)[condition]
    target_val_data = np.array(val_data)[condition]

    #save the number of target data point to use to balance the non-target data
    num_balanced_samples = len(target_val_data) 
    print("Num of Target Samples: ",num_balanced_samples)
    condition = np.array(new_val_label) == len(target_family)
    excess_new_val_label = np.array(new_val_label)[condition]
    excess_val_data = np.array(val_data)[condition]

    random_choice_indices = np.random.choice(
            len(excess_new_val_label),
            size=num_balanced_samples, 
            replace=False
        )

    print("Length random_choice_indices: ", len(random_choice_indices))

    excess_new_val_label = excess_new_val_label[random_choice_indices]
    excess_val_data = excess_val_data[random_choice_indices]

    val_data = np.concatenate((target_val_data,excess_val_data), axis = 0)
    new_val_label = np.concatenate((target_new_val_label,excess_new_val_label), axis = 0)


    print("length balanced train data: ", len(val_data))
    print("length balanced train data labels: ", len(new_val_label))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time

class Timer:
    """A simple context manager for timing code blocks."""
    def __init__(self, timings_dict, key):
        self.timings = timings_dict
        self.key = key
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.key}] starting...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.timings[self.key] = elapsed
        print(f"[{self.key}] finished in {elapsed:.2f} seconds.")

def report_timings(timings):
    """Prints a summary of the timings."""
    print("\n--- Timing Report ---")
    total_time = 0
    for key, value in timings.items():
        print(f"  - {key}: {value:.2f} seconds")
        total_time += value
    print("---------------------")
    print(f"  Total Elapsed Time: {total_time:.2f} seconds")

def save_report(report_path, config, model_summary, history, final_accuracy, timings):
    """Saves a detailed report of the training session to a text file."""
    print(f"\n--- Saving Training Report to {report_path} ---")
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("           TRAINING SESSION REPORT\n")
        f.write("="*60 + "\n\n")
        
        # --- 1. Configuration ---
        f.write("--- 1. Training Configuration ---\n")
        for key, value in config.items():
            f.write(f"  - {key}: {value}\n")
        f.write("\n")

        # --- 2. Model Architecture ---
        f.write("--- 2. Model Architecture ---\n")
        f.write(model_summary + "\n\n")

        # --- 3. Training History ---
        f.write("--- 3. Training History (Epoch-wise) ---\n")
        epochs = range(1, len(history['train_loss']) + 1)
        header = f"{'Epoch':<7} | {'Train Loss':<12} | {'Train Acc (%)':<15} | {'Val Acc (%)':<13}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for i, epoch in enumerate(epochs):
            loss = history['train_loss'][i]
            train_acc = history['train_acc'][i]
            val_acc = history['val_acc'][i]
            f.write(f"{epoch:<7} | {loss:<12.4f} | {train_acc:<15.2f} | {val_acc:<13.2f}\n")
        f.write("\n")

        # --- 4. Final Performance ---
        f.write("--- 4. Final Performance on Test Set ---\n")
        f.write(f"  - Accuracy: {final_accuracy:.2f} %\n\n")

        # --- 5. Timing Report ---
        f.write("--- 5. Timing Report ---\n")
        total_time = 0
        for key, value in timings.items():
            f.write(f"  - {key}: {value:.2f} seconds\n")
            total_time += value
        f.write("---------------------\n")
        f.write(f"  Total Elapsed Time: {total_time:.2f} seconds\n")
    
    print("Report saved successfully.")

class DynamicCNN(nn.Module):
    """
    A dynamically configurable Convolutional Neural Network (CNN).
    The user can specify the architecture including convolutional layers,
    pooling layers, and fully connected layers.
    """
    def __init__(self, input_channels, num_classes, conv_layers_config, fc_layers_config, num_iq_samples):
        """
        Initializes the DynamicCNN model.

        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB images).
            num_classes (int): Number of output classes.
            conv_layers_config (list of tuples): Each tuple defines a convolutional block.
                Format: (out_channels, kernel_size, stride, padding, use_pooling)
                'use_pooling' is a boolean to add a MaxPool2d layer after the conv layer.
            fc_layers_config (list of int): A list where each integer is the number of
                neurons in a fully connected layer.
        """
        super(DynamicCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        in_channels = input_channels

        print("model intial in_channels: ", in_channels)

        # Create convolutional layers dynamically
        for out_channels, kernel_size, stride, padding, use_pooling in conv_layers_config:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            if use_pooling:
                self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.pool_layers.append(nn.Identity()) # Placeholder if no pooling
            in_channels = out_channels

        # To determine the input size of the first fully connected layer,
        # we need to do a forward pass with a dummy tensor.
        # This is a common practice when the architecture is dynamic.
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, num_iq_samples[0], num_iq_samples[1]) # !!! Changeing this to account for shape of IQ data
            dummy_output = self._forward_conv(dummy_input)
            flattened_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        in_features = flattened_size

        print("in features for first FC layer: ", in_features)

        # Create fully connected layers dynamically
        for out_features in fc_layers_config:
            self.fc_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        
        # Output layer
        self.output_layer = nn.Linear(in_features, num_classes)

    def _forward_conv(self, x):
        """Helper function to forward through conv layers only."""
        count = 0
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            '''
            count+=1
            print("layer: ", count, " with input len", len(x), "and input: ", x)
            print("x[0]: ", x[0]) 
            print("len x[0]: ", len(x[0]))
            print("x[0][0]: ", x[0][0]) 
            print("len x[0][0]: ", len(x[0][0])) 
            '''
            x = torch.relu(conv(x))
            x = pool(x)
        return x

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        '''
        print("start x: ", x) 
        print("len start x: ", len(x)) 
        print("start x[0]: ", x[0]) 
        print("len start x[0]: ", len(x[0]))
        print("start x[0][0]: ", x[0][0]) 
        print("len start x[0][0]: ", len(x[0][0])) 
        '''
        x = self._forward_conv(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Forward pass through fully connected layers
        for fc_layer in self.fc_layers:
            x = torch.relu(fc_layer(x))
            
        x = self.output_layer(x)
        return x



def evaluate_model(model, loader, device='cpu'):
    """
    Evaluates the model's accuracy on a given dataset.
    Used for validation and training accuracy checks.
    """
    model.to(device)
    model.eval() # Set the model to evaluation mode
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    Trains the PyTorch model and tracks history for plotting.
    """
    model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    total_training_time = 0
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train() # Set the model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        
        # Evaluate on training and validation set after each epoch
        train_acc = evaluate_model(model, train_loader, device)
        val_acc = evaluate_model(model, val_loader, device)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_duration = time.time() - epoch_start_time
        total_training_time += epoch_duration
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% -- Took {epoch_duration:.2f}s')
            
    print(f"Finished training. Total training time: {total_training_time:.2f}s")
    return history, total_training_time

def test_model(model, test_loader, device='cpu'):
    """
    Tests the PyTorch model.
    """
    model.to(device)
    model.eval() # Set the model to evaluation mode
    
    correct = 0
    total = 0
    
    print("Starting testing...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f} %')
    return accuracy

def plot_results(history, root):
    """
    Plots training and validation metrics using matplotlib.
    """
    print("\n--- Plotting Results ---")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found. Please install it ('pip install matplotlib') to plot results.")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 6))

    # Plot Training & Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'o-', label='Training Accuracy', color='b')
    plt.plot(epochs, history['val_acc'], 'o-', label='Validation Accuracy', color='r')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Training Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'o-', label='Training Loss', color='g')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(root+'training_results.png')
    print("Results plot saved as 'training_results.png'")
    # You can uncomment the line below to display the plot directly
    # plt.show()

def save_model(model, path="cnn_model.pth"):
    """
    Saves the trained model's state dictionary.
    Handles models wrapped in nn.DataParallel.
    """
    print(f"Saving model to {path}...")
    # If the model is wrapped in DataParallel, save the underlying model's state_dict
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    print("Model saved.")

def load_model(model_class, path, *args, **kwargs):
    """
    Loads a model's state dictionary from a file.
    Args:
        model_class: The class of the model to be instantiated.
        path (str): The path to the saved model file.
        *args, **kwargs: Arguments needed to instantiate the model class.
    """
    print(f"Loading model from {path}...")
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    print("Model loaded.")
    return model

def analyze_dataset(loader, dataset_name="Training"):
    """
    Performs and prints a statistical analysis on the dataset from a DataLoader.
    Calculates class distribution, image dimensions, and pixel value statistics.

    Args:
        loader (DataLoader): The DataLoader for the dataset to analyze.
        dataset_name (str): Name of the dataset (e.g., "Training", "Testing").
    """
    print(f"\n--- Analyzing {dataset_name} Dataset ---")

    if not hasattr(loader, 'dataset'):
        print("DataLoader does not have a 'dataset' attribute.")
        return

    dataset = loader.dataset
    
    # Handle datasets wrapped in torch.utils.data.Subset (from random_split)
    if isinstance(dataset, torch.utils.data.Subset):
        # Get labels and data from the original dataset using subset indices
        labels = [dataset.dataset.tensors[1][i] for i in dataset.indices]
        data_tensor = torch.stack([dataset.dataset.tensors[0][i] for i in dataset.indices])
        labels_tensor = torch.tensor(labels)
    elif isinstance(dataset, TensorDataset):
        # Handle regular TensorDataset
        data_tensor = dataset.tensors[0]
        labels_tensor = dataset.tensors[1]
    else:
        print(f"Dataset type {type(dataset).__name__} is not supported for analysis.")
        return

    if len(labels_tensor) == 0:
        print("No labels found to analyze.")
        return

    # 1. Class Distribution
    print("Class Distribution:")
    unique_classes, counts = torch.unique(labels_tensor, return_counts=True)
    class_dist = dict(zip(unique_classes.tolist(), counts.tolist()))
    
    for cls in sorted(class_dist.keys()):
        percentage = (class_dist[cls] / len(labels_tensor)) * 100
        print(f"  Class {cls}: {class_dist[cls]} samples ({percentage:.2f}%)")

    # 2. General Metrics
    num_samples = data_tensor.shape[0]
    print(f"\nTotal Samples: {num_samples}")

    if data_tensor.dim() == 4:  # Expected format: (N, C, H, W)
        num_channels = data_tensor.shape[1]
        height = data_tensor.shape[2]
        width = data_tensor.shape[3]
        print(f"Image Dimensions: {height}x{width}")
        print(f"Number of Channels: {num_channels}")

        # 3. Mean and Standard Deviation of pixel values per channel
        # This is useful for data normalization
        print("\nPixel Value Statistics (per channel):")
        # Reshape to (N*H*W, C) to calculate stats per channel easily
        pixels = data_tensor.permute(0, 2, 3, 1).reshape(-1, num_channels)
        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0)

        for c in range(num_channels):
            print(f"  Channel {c}: Mean={mean[c]:.4f}, Std Dev={std[c]:.4f}")
    
    print("--------------------------------------")


def create_dummy_dataset(num_samples=1000, img_size=(64, 64), num_classes=10, channels=3):
    """
    Creates a dummy dataset for demonstration purposes.
    Returns:
        (DataLoader, DataLoader): train_loader, test_loader
    """
    print("Creating a dummy dataset...")
    # Generate random images and labels
    data = np.random.rand(num_samples, channels, img_size[0], img_size[1]).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples).astype(np.int64)

    # Convert to PyTorch tensors
    tensor_x = torch.Tensor(data)
    tensor_y = torch.Tensor(labels).long()

    dataset = TensorDataset(tensor_x, tensor_y)
    
    # Split into training and testing
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Dummy dataset created.")
    return train_loader, test_loader

def Format_RF_dataset(data, labels, batch_size = 32, train_test_split = 0.2):
    """
    Formats labeled IQ data into dataloader
    Returns:
        (DataLoader, DataLoader): train_loader, test_loader
    """
    print("Formatting RF dataset...")
    # Generate random images and labels
    #data = np.random.rand(num_samples, channels, img_size[0], img_size[1]).astype(np.float32)
    #labels = np.random.randint(0, num_classes, num_samples).astype(np.int64)

    data = np.array(data)
    labels = np.array(labels)

    # Convert to PyTorch tensors
    tensor_x = torch.Tensor(data)
    tensor_y = torch.Tensor(labels).long()

    dataset = TensorDataset(tensor_x, tensor_y)
    
    # Split into training and testing
    train_size = int((1-train_test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("RF dataset loader created.")
    return train_loader, test_loader


#74% accuracy with output_channels = 16, batch size 32, lr = 0.0001 for PSKvAMvFM, 800 samples each, epoch = 30
#77% w output_channels = 64 , batch_size = 8, epoch = 50
#74% @ epochs = 100, 323.327 MB
#67% w output_channels = 128, batch = 8, epoch = 30, 1293.217 MB
#73% w output_channels = 256, batch = 8, epoch = 30, 5172.685 MB
### All above with one FC layer = [output_channels*2*2*2*2, output_channels]
#NOTE: FC layer should not scale 1:1 with the size of CNN layers, it quickly becomes too large
#75.5% w output_channels = 256, batch_size = 32, epoch = 30, fc_config = [output_channels*2*2, output_channels], 3249.673 MB
#output channel above 256 seems bad and slow
#81.18 % test accuracy, 100% train accuracy, 500 epoch, same config as above




print(f"num classes {num_classes}")
print(f"input length {num_iq_samples}")
print(f"output channels {output_channels}")


# --- Configuration ---
config = {
    'INPUT_CHANNELS': 1, #### going to try treating IQ as one input channel of 2d data input size *2
    'NUM_CLASSES': num_classes,
    'IMAGE_SIZE': (2,num_iq_samples),
    'DEVICE': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    'MODEL_SAVE_PATH': root+"/"+target_family_string+"_custom_cnn.pth",
    'REPORT_SAVE_PATH': root+"/"+target_family_string+"_training_report.txt",
    'NUM_EPOCHS': epochs,
    'LEARNING_RATE': lr #good learning rates: 0.0001
}

# Define a custom CNN architecture
# Format: (out_channels, kernel_size, stride, padding, use_pooling)

'''
conv_config = [
    (output_channels, 2, 1, 0, True),   # 1st Conv Layer -> MaxPool ##!!!removed padding ###!!!set kernel size to 1
    (output_channels*2, 2, 1, 0, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2, 2, 1, 0, False)   # 3rd Conv Layer
]
'''

'''
##!!!!This one works!!!!
conv_config = [
    (16, 2, 1, 1, True),   # 1st Conv Layer -> MaxPool
    (32, 2, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (64, 2, 1, 1, False)   # 3rd Conv Layer
]
'''




'''
conv_config = [
    (output_channels, 2, 2, 1, True),   # 1st Conv Layer -> MaxPool
    (output_channels*2*2, 2, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2, 2, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2, 2, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 1, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 2, 1, 1, True),   # 2nd Conv Layer -> MaxPool
    (output_channels*2*2*2*2*2, 2, 1, 1, False),   # 2nd Conv Layer -> MaxPool
]
'''




# --- 1. Create a dummy dataset ---
# In a real scenario, you would replace this with your own data loading logic.
# Ensure your data is in the format of (N, C, H, W)
# N: Number of samples, C: Channels, H: Height, W: Width

timings = {} # Dictionary to store timings of each step

with Timer(timings, "1. Dataset Creation & Analysis"):
    train_loader, test_loader= Format_RF_dataset(train_data, new_train_label, batch_size = batch_size, train_test_split = 0.2)
    val_loader,_ = Format_RF_dataset(val_data, new_val_label, batch_size = batch_size, train_test_split = 0.0)#train test split set to 0 for val set
    # --- 1.5. Analyze the training data ---
    analyze_dataset(train_loader, dataset_name="Training")

# --- 1.6. Create a validation split ---
with Timer(timings, "2. Validation Split"):
    # In a real-world scenario, you would have a separate validation dataset.
    # Here, we split the original training data for demonstration.
    print("\n--- Creating Validation Split ---")
    #train_dataset = train_loader.dataset
    #val_split = 0.2
    #dataset_size = len(train_dataset)
    #val_size = int(val_split * dataset_size)
    #train_size = dataset_size - val_size
    
    #new_train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create new DataLoaders for the split data
    #new_train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    #val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    print(f"Split training data into {len(train_data)} training samples and {len(val_data)} validation samples.")




# --- 2. Create the CNN model ---
with Timer(timings, "3. Model Creation"):
    print("\n--- Creating Model ---")
    my_cnn = DynamicCNN(
        input_channels=config['INPUT_CHANNELS'], 
        num_classes=config['NUM_CLASSES'],
        conv_layers_config=conv_config,
        fc_layers_config=fc_config,
        num_iq_samples = config['IMAGE_SIZE']
    )
    
    # Check for multiple GPUs and wrap the model with DataParallel for multi-GPU training
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        my_cnn = nn.DataParallel(my_cnn)

    model_summary_str = str(my_cnn)
    print("Model Architecture:")
    print(model_summary_str)

# --- 3. Define Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_cnn.parameters(), lr=config['LEARNING_RATE'])

param_size = sum(param.nelement() * param.element_size() for param in my_cnn.parameters())


# Calculate buffer size
buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in my_cnn.buffers())

# Total size in MB
total_size_mb = (param_size + buffer_size) / 1024**2
print(f"Model size: {total_size_mb:.3f} MB")

# --- 4. Train the model ---
print("\n--- Training Model ---")
history, training_time = train_model(
    my_cnn, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    num_epochs=config['NUM_EPOCHS'], 
    device=config['DEVICE']
)
timings["4. Model Training"] = training_time


# --- 5. Plot training results ---
with Timer(timings, "5. Plotting Results"):
    file_prefix = root+"/"+target_family_string
    plot_results(history, file_prefix)

# --- 6. Test the model on the unseen test set ---
with Timer(timings, "6. Final Model Testing"):
    print("\n--- Testing Model ---")
    final_test_accuracy = test_model(my_cnn, test_loader, device=config['DEVICE'])

# --- 7. Save the model ---
with Timer(timings, "7. Saving Model"):
    print("\n--- Saving Model ---")
    save_model(my_cnn, path=config['MODEL_SAVE_PATH'])

# --- 8. Load the model ---
with Timer(timings, "8. Loading Model"):
    print("\n--- Loading Model ---")
    # We need to provide the same configuration to instantiate the model class
    # before loading the saved weights.
    loaded_model = load_model(
        DynamicCNN,
        path=config['MODEL_SAVE_PATH'],
        input_channels=config['INPUT_CHANNELS'],
        num_classes=config['NUM_CLASSES'],
        conv_layers_config=conv_config,
        fc_layers_config=fc_config,
        num_iq_samples = config['IMAGE_SIZE']
    )
    print("Loaded Model Architecture:")
    print(loaded_model)

# --- 9. Test the loaded model to verify it's working ---
with Timer(timings, "9. Testing Loaded Model"):
    print("\n--- Testing Loaded Model ---")
    test_model(loaded_model, test_loader, device=config['DEVICE'])

# --- 10. Report Timings ---
report_timings(timings)
# Calculate parameter size
param_size = sum(param.nelement() * param.element_size() for param in loaded_model.parameters())


# Calculate buffer size
buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in loaded_model.buffers())

# Total size in MB
total_size_mb = (param_size + buffer_size) / 1024**2
print(f"Model size: {total_size_mb:.3f} MB")

output_string = ""

output_string+=(f"Creating dataset with fft_size {fft_size}")
output_string+="\n"
output_string+=(f" IQ samples per datapoint {num_iq_samples_dataset}")
output_string+="\n"
output_string+=(f" Samples per modulation class {num_samples_per_class}")
output_string+="\n"
output_string+=(f" Percent used for validation {percent_validation}")
output_string+="\n"
output_string+=(f" Target modulations {target_family}")
output_string+="\n"
output_string+=(f"Conv Config {conv_config}")
output_string+="\n"
output_string+=(f"FNN Config {fc_config}")
output_string+="\n"
output_string+= (f"Modulation Class list {class_list}")

model_summary_str = model_summary_str +"\n"+(f"Model size: {total_size_mb:.3f} MB")+"\n"+output_string

# --- 11. Save Final Report ---
with Timer(timings, "10. Saving Report"):
    save_report(
        report_path=config['REPORT_SAVE_PATH'],
        config=config,
        model_summary=model_summary_str,
        history=history,
        final_accuracy=final_test_accuracy,
        timings=timings
    )


if del_dataset_when_complete:
    ##Delete the generated dataset files to save storage space
    # Specify the path to the file you want to delete
    file_to_delete = f"{root}/train/data.h5"

    # Check if the file exists before attempting to delete it
    if os.path.exists(file_to_delete):
        try:
            os.remove(file_to_delete)
            print(f"File '{file_to_delete}' deleted successfully.")
        except PermissionError:
            print(f"Permission denied: Unable to delete '{file_to_delete}'.")
        except Exception as e:
            print(f"An error occurred while deleting '{file_to_delete}': {e}")
    else:
        print(f"File '{file_to_delete}' does not exist.")

        # Specify the path to the file you want to delete
    file_to_delete = f"{root}/val/data.h5"

    # Check if the file exists before attempting to delete it
    if os.path.exists(file_to_delete):
        try:
            os.remove(file_to_delete)
            print(f"File '{file_to_delete}' deleted successfully.")
        except PermissionError:
            print(f"Permission denied: Unable to delete '{file_to_delete}'.")
        except Exception as e:
            print(f"An error occurred while deleting '{file_to_delete}': {e}")
    else:
        print(f"File '{file_to_delete}' does not exist.")




print("!"*50)
print("Model Training Complete")
print("!"*50)
