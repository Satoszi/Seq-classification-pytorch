import pytest
import utils
import numpy as np
import itertools

from torch.utils.data import DataLoader 

def get_data_loader(avg_seq, min_seq_length, max_seq_length):

    custom_train_loader = utils.MnistSequences(
        batch_size=8, 
        avg_seq = avg_seq, 
        min_seq_length = min_seq_length, 
        max_seq_length = max_seq_length
    )

    data_loader = DataLoader(
        dataset=custom_train_loader, 
        batch_size=8, 
        shuffle=True
    )

    return data_loader
    
def check_sequence_distribution(avg_seq, min_seq_length, max_seq_length):

    data_loader = get_data_loader(avg_seq, min_seq_length, max_seq_length)
    data_batches = itertools.islice(data_loader, 0, 20)
    x_data_batches = list(map(lambda x: x[0], data_batches))
    multi_batch_single_sequence = list(map(lambda x: x[0], x_data_batches))
    sequences_lengths = list(map(lambda x: len(x), multi_batch_single_sequence))
    
    seq_length_range = np.max(sequences_lengths) - np.min(sequences_lengths)
    
    assert seq_length_range >= 1
    assert np.max(sequences_lengths) <= max_seq_length
    assert np.min(sequences_lengths) >= min_seq_length

def test_sequence_distribution():
    
    check_sequence_distribution(10., 3,30)
    check_sequence_distribution(10., 5,15)
    check_sequence_distribution(5., 3,8)
    check_sequence_distribution(35., 25,35)


