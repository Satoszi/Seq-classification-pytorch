import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

class ValueOutOfRangeError(Exception):
    pass
    
class MnistSequences(Dataset):
    
    def __init__(
        self, 
        train=True, 
        min_seq_length = 3, 
        max_seq_length = 30, 
        avg_seq = 10., 
        std_seq = 3., 
        batch_size = 64, 
        weak_supervision_rate = 0,
        searched_digit = 4,
    ):
        
        if weak_supervision_rate > 1 or weak_supervision_rate < 0:
            raise ValueOutOfRangeError
        
        self.dataset = datasets.MNIST(
            root="dataset/", 
            train=train, 
            transform=transforms.ToTensor(), 
            download=True
        )
        
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.avg_seq = avg_seq
        self.std_seq = std_seq
        self.batch_size = batch_size
        self.weak_supervision_rate = weak_supervision_rate
        
        self.iterations_counter = 0
        self.sequence_length = self.get_random_sequence_length()
        self.searched_digit = searched_digit

    def __len__(self):
          return len(self.dataset)

    def reset(self):
        self.iterations_counter = 0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_weak_supervision_rate(self, weak_supervision_rate):
        if weak_supervision_rate > 1 or weak_supervision_rate < 0:
            raise ValueOutOfRangeError
        self.weak_supervision_rate = weak_supervision_rate

    def get_random_sequence_length(self):
        sequence_length = torch.normal(mean=torch.tensor(self.avg_seq), std=torch.tensor(self.std_seq))
        sequence_length = sequence_length.type(torch.int)

        if (sequence_length < self.min_seq_length): return torch.tensor(self.min_seq_length)
        if (sequence_length > self.max_seq_length): return torch.tensor(self.max_seq_length)
        return sequence_length
        
    def __getitem__(self, index):

        # Get new sequence length every new batch,
        # but keep the same length within the same batch
        if (self.iterations_counter % self.batch_size == 0): 
            self.sequence_length = self.get_random_sequence_length()
        self.iterations_counter += 1
        
        index = np.clip(index,0,self.__len__()-self.sequence_length-1)
        x_sequence_point = [self.dataset[i][0] for i in range(index,(index + self.sequence_length))]
        x_sequence_point = torch.cat(x_sequence_point)
        x_sequence_point = torch.reshape(x_sequence_point, (self.sequence_length, 28,28))

        y_mnist_labels = [self.dataset[i][1] for i in range(index,(index + self.sequence_length))]
        y_sequence_label = 0
        if self.searched_digit in y_mnist_labels: 
            y_sequence_label = 1
        
        # weak supervision scenerio
        if np.random.random() < self.weak_supervision_rate:
            y_sequence_label = 1 - y_sequence_label
        
        return x_sequence_point, y_sequence_label
        
        
def validate(data_loader, model, batches_number, device):
    N = 0
    num_correct = 0
    
    # Set model to evaluate state
    model.eval()

    with torch.no_grad():
        for x_data, y_data in itertools.islice(data_loader, 0, batches_number):
            
            x_data = x_data.to(device=device).squeeze(1)
            y_data = y_data.to(device=device)

            predictions = model(x_data)
            _, predictions = predictions.max(1)
            num_correct += (predictions == y_data).sum()
            N += predictions.size(0)

    return float(num_correct / N)
    
    
def save_fig(history, fig_name):
    validation_history = [i[0] for i in history]
    train_history = [i[1] for i in history]
    
    plt.figure(1, figsize = (10, 7)) 
    
    plt.plot(train_history)  
    plt.plot(validation_history)  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('batches*n')  
    plt.legend(['train', 'valid']) 
    plt.grid()
    
    path = "results//"+fig_name
    plt.savefig(path, bbox_inches='tight')
    plt.close()