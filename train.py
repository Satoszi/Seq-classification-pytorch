import torch
from torch import optim 
from torch import nn  
from torch.utils.data import DataLoader 
from torchsummary import summary

from tqdm import tqdm 
import os

import utils
import nn_model
import config

def monitor(
    train_loader, 
    test_loader, 
    model,
    batch_idx, 
    wsr,
    device
):

    # Validate
    validation_acc = utils.validate(test_loader, model, 20, device)
    train_acc = utils.validate(train_loader, model, 20, device)
    print("Val acc:",validation_acc, "Train acc:", train_acc)   

    # Save checkpoint
    save_name = f"batch_idx_{wsr}_{batch_idx}.pt"
    checkpoint = os.path.join("checkpoints",save_name)
    torch.save(model, checkpoint)

    return validation_acc,train_acc

class DatasetGenerator():
    def __init__(self, weak_supervision_rate = 0) -> None:
        
        self.weak_supervision_rate = weak_supervision_rate
        self.custom_train_dataset = utils.MnistSequences(weak_supervision_rate = self.weak_supervision_rate)
        self.custom_test_dataset = utils.MnistSequences(train=False)

        self.train_loader = DataLoader(
            dataset=self.custom_train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True
        )

        self.test_loader = DataLoader(
            dataset=self.custom_test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True
        )

    def reset_loaders(self):
        self.custom_train_dataset.reset()
        self.custom_test_dataset.reset()

        self.custom_train_dataset.set_weak_supervision_rate(self.weak_supervision_rate)

        
        self.train_loader = DataLoader(
            dataset=self.custom_train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True
        )

        self.test_loader = DataLoader(
            dataset=self.custom_test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True
        )

    def set_weak_supervision_rate(self, weak_supervision_rate):
        self.weak_supervision_rate = weak_supervision_rate

    def get_test_loader(self):
        return self.test_loader

    def get_train_loader(self):
        return self.train_loader


def main():

    loader = DatasetGenerator()
    train_loader = loader.get_train_loader()
    test_loader = loader.get_test_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Active device:", device)

    for wsr in config.WEAK_SUPERVISION_RATES:

        model = nn_model.LSTM_Model(
            config.INPUT_SIZE, 
            config.LSTM_HIDDEN_SIZE, 
            config.NUM_LSTM_LAYERS, 
            device
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        summary(model)
        # Set model to train state
        model.train()

        history = []

        loader.set_weak_supervision_rate(wsr)
        loader.reset_loaders()
        train_loader = loader.get_train_loader()
        test_loader = loader.get_test_loader()

        for _ in range(config.NUM_EPOCHS):
            
            for batch_idx, (x_data, y_data) in enumerate(tqdm(train_loader)):

                x_data = x_data.to(device=device).squeeze(1)
                y_data = y_data.to(device=device)

                # forward
                outputs = model(x_data)
                loss = criterion(outputs, y_data)

                # backward
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                
                # Monitor
                if batch_idx % 50 == 0:
                    results = monitor(
                        train_loader, 
                        test_loader, 
                        model,
                        batch_idx, 
                        wsr,
                        device,
                    )
                    history.append(results)

        utils.save_fig(history, f"training_history_wsr_{wsr}.png")

if __name__ == "__main__":
    main()