import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np

class MnistSequences(Dataset):
    
    def __init__(self, train=True, min_seq_length = 3, max_seq_length = 30, avg_seq = 10., 
    std_seq = 3., batch_size = 64):
    
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.avg_seq = avg_seq
        self.std_seq = std_seq
        self.dataset = datasets.MNIST(root="dataset/", train=train, transform=transforms.ToTensor(), download=True)
        self.iterations_counter = 0
        self.sequence_length = self.get_random_sequence_length()
        self.batch_size = batch_size
    def get_random_sequence_length(self):
        sequence_length = torch.normal(mean=torch.tensor(self.avg_seq), std=torch.tensor(self.std_seq))
        sequence_length = sequence_length.type(torch.int)

        if (sequence_length < self.min_seq_length): return torch.tensor(self.min_seq_length)
        if (sequence_length > self.max_seq_length): return torch.tensor(self.max_seq_length)
        return sequence_length
        
    def reset(self):
        self.iterations_counter = 0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def __len__(self):
          return len(self.dataset)

    def __getitem__(self, index):

        if (self.iterations_counter % self.batch_size == 0): self.sequence_length = self.get_random_sequence_length()

        self.iterations_counter += 1
        index = np.clip(index,0,self.__len__()-self.sequence_length-1)
        
        x_point = [self.dataset[i][0] for i in range(index,(index + self.sequence_length))]
        x_point = torch.cat(x_point)
        x_point = torch.reshape(x_point, (self.sequence_length, 28,28))

        y_labels = [self.dataset[i][1] for i in range(index,(index + self.sequence_length))]
        y_point = 0
        if 4 in y_labels: y_point = 1

        return x_point, y_point