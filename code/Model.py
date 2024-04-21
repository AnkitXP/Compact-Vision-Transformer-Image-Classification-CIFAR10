import torch
from torch import nn
import os
import numpy as np
from Network import RelativeViT
from DataLoader import custom_dataloader
import timeit
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = RelativeViT(configs).to('cuda')
      
    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):

        start = timeit.default_timer()

        train_dataloader = custom_dataloader(x_train, y_train, batch_size=configs.batch_size, train=True)
        val_dataloader = custom_dataloader(x_valid, y_valid, batch_size=configs.batch_size, train=True)

        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                          lr = configs.learning_rate, 
                                          betas = configs.betas, 
                                          weight_decay = configs.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=configs.learning_rate,
                                             steps_per_epoch=len(train_dataloader), epochs=configs.num_epochs)

        for epoch in tqdm(range(1, configs.num_epochs + 1), position=0, leave=True):

            self.network.train()

            train_labels = []
            train_preds = []
            train_running_loss = 0

            for idx, (images, labels) in enumerate(tqdm(train_dataloader, position=0, leave=True)):

                current_images = torch.tensor(images, dtype=torch.float32).to('cuda')
                current_labels = torch.tensor(labels, dtype=torch.int64).to('cuda')

                predictions = self.network(current_images)
                prediction_labels = torch.argmax(predictions, dim=1)

                loss = self.cross_entropy_loss(predictions, current_labels)

                train_labels.extend(labels.cpu().detach())
                train_preds.extend(prediction_labels.cpu().detach())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_running_loss += loss.item()
            
            train_loss = train_running_loss / (idx +1)

            self.network.eval()
            
            val_labels = []
            val_preds = []
            val_running_loss = 0

            with torch.no_grad():
                for idx, (validation_images, validation_labels) in enumerate(tqdm(val_dataloader, position=0, leave=True)):

                    validation_images = torch.tensor(validation_images, dtype=torch.float32).to('cuda')
                    validation_labels = torch.tensor(validation_labels, dtype=torch.int64).to('cuda')

                    val_predictions = self.network(validation_images)
                    val_prediction_labels = torch.argmax(val_predictions, dim=1)

                    val_labels.extend(validation_labels.cpu().detach())
                    val_preds.extend(val_prediction_labels.cpu().detach())

                    loss = self.cross_entropy_loss(val_predictions, validation_labels)
                    val_running_loss += loss.item()
            val_loss = val_running_loss / (idx + 1)

            print("-"*30)
            print(f"EPOCH {epoch}: Train Loss {train_loss:.4f}, Valid Loss {val_loss:.4f}")
            print(f"EPOCH {epoch}: Train Accuracy {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}, Valid Accuracy {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
            print("-"*30)

            if (epoch) % configs.save_interval == 0:
                self.save(epoch)

        stop = timeit.default_timer()
        print(f"Training Time: {stop - start : .2f}s")

    def evaluate(self, x, y):

        test_dataloader = custom_dataloader(x, y, batch_size=128, train=False)

        self.network.eval()
            
        test_labels_final = []
        test_preds_final = []

        with torch.no_grad():
            for idx, (testing_images, testing_labels) in enumerate(tqdm(test_dataloader, position=0, leave=True)):

                test_images = torch.tensor(testing_images, dtype=torch.float32).to('cuda')
                test_labels = torch.tensor(testing_labels, dtype=torch.int64).to('cuda')

                test_predictions = self.network(test_images)
                test_prediction_labels = torch.argmax(test_predictions, dim=1)

                test_labels_final.extend(test_labels.cpu().detach())
                test_preds_final.extend(test_prediction_labels.cpu().detach())

        print(f"Test Accuracy: {np.sum(np.array(test_preds_final) == np.array(test_labels_final))/len(test_labels_final):.2f}")

    def predict_prob(self, x):
        x = x.reshape(-1, 3, 32, 32)
        predict_dataloader = custom_dataloader(x, x, batch_size=128, train=False)
        self.network.eval()
        
        predict_proba_final = []
        
        with torch.no_grad():
            for idx, (prediction_images, _) in enumerate(tqdm(predict_dataloader, position=0, leave=True)):
                test_data = torch.tensor(prediction_images, dtype=torch.float32).to('cuda')
                probabilities = self.network(test_data)
                predict_proba_final.extend(probabilities.cpu().detach())

        return np.stack(predict_proba_final, axis=0)

    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs.save_dir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.configs.save_dir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))