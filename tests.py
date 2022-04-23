import pytest
import utils
import numpy as np

from torch.utils.data import DataLoader 

custom_train_loader = utils.MnistSequences(batch_size=8)
train_loader = DataLoader(dataset=custom_train_loader, batch_size=8, shuffle=True)

def test_sequence_length():

    sequences_lengths = []
    for idx, (x, y) in enumerate(train_loader):
        sequences_lengths.append(len(x[0]))
        if idx > 10: break
        
    assert np.max(sequences_lengths) - np.min(sequences_lengths) > 5

test_sequence_length()