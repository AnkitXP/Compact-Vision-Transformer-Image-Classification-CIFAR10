import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2
import os
import numpy as np
from Network import CompactVisionTransformer
from DataLoader import custom_dataloader
import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import math

import warnings
warnings.filterwarnings('ignore')

import gc
import sys

"""This script defines the training, validation and testing process.
"""

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = CompactVisionTransformer(configs).to('cuda')
        cutmix = v2.CutMix(num_classes=configs.num_classes)
        mixup = v2.MixUp(num_classes=configs.num_classes)
        self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
      
    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):

        print("<===================================================================== Training =====================================================================>")

        start = timeit.default_timer()

        train_dataloader = custom_dataloader(x_train, y_train, batch_size=configs.batch_size, train=True)
        val_dataloader = custom_dataloader(x_valid, y_valid, batch_size=configs.batch_size, train=True)


        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                          lr = configs.learning_rate,  
                                          weight_decay = configs.weight_decay)
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr=configs.learning_rate, momentum=0.9, weight_decay=0.0001)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=configs.num_epochs, eta_min = 1e-5)
        self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps = 5, t_total=configs.num_epochs)
        
        train_loss_history = []
        val_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []
        learning_rate_history = []

        for epoch in tqdm(range(1, configs.num_epochs + 1), position=0, leave=True):

            self.network.train()

            train_labels = []
            train_preds = []
            train_running_loss = 0

            for idx, (images, labels) in enumerate(tqdm(train_dataloader, position=0, leave=True)):

                cutmix_images, cutmix_labels = self.cutmix_or_mixup(images, labels)

                # print(cutmix_labels[0])
                # sys.exit(0)

                current_images = torch.tensor(cutmix_images, dtype=torch.float32).to('cuda')
                current_labels = torch.tensor(cutmix_labels, dtype=torch.float32).to('cuda')

                # if idx == 0:
                #     visualize(images[0].detach().numpy(), '../results/train.png')

                self.optimizer.zero_grad()

                predictions = self.network(current_images)
                loss = self.cross_entropy_loss(predictions, current_labels)

                predicted = torch.argmax(predictions, -1)
                labels = torch.argmax(current_labels, -1)

                train_labels.extend(labels.cpu().detach())
                train_preds.extend(predicted.cpu().detach())

                loss.backward()
                self.optimizer.step()

                train_running_loss += loss.item()
            
            train_loss = train_running_loss / (idx +1)
            train_loss_history.append(train_loss)

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
            val_loss_history.append(val_loss)

            train_accuracy = sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels)
            valid_accuracy = sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels)

            train_accuracy_history.append(train_accuracy)
            val_accuracy_history.append(valid_accuracy)
            learning_rate_history.append(self.scheduler.get_last_lr())

            print("-"*30)
            print(f"EPOCH {epoch}: Train Loss {train_loss:.4f}, Valid Loss {val_loss:.4f}")
            print(f"EPOCH {epoch}: Train Accuracy {train_accuracy:.4f}, Valid Accuracy {valid_accuracy:.4f}")
            print("-"*30)
            self.plot_metrics(self.configs.result_dir, train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history, learning_rate_history)
            
            self.scheduler.step()

            if (epoch) % configs.save_interval == 0:
                self.save(configs)

        stop = timeit.default_timer()
        print(f"Training Time: {stop - start : .2f}s")
        

    def evaluate(self, x, y):

        print("<===================================================================== Testing =====================================================================>")

        test_dataloader = custom_dataloader(x, y, batch_size=128, train=False)
            
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
        print("<===================================================================== Prediction =====================================================================>")

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
        checkpoint_path = os.path.join(self.configs.save_dir, 'cifar10-classifier-CCT_7x4_2-CutMix-warmup-1.ckpt')
        os.makedirs(self.configs.save_dir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def plot_metrics(self, result_dir, train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history, learning_rate_history):
        
        print(result_dir)
        os.makedirs(result_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(result_dir, 'loss_plot-warm-5.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracy_history, label='Train Accuracy', color='blue')
        plt.plot(val_accuracy_history, label='Validation Accuracy', color='orange')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(result_dir, 'accuracy_plot-warm-5.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(learning_rate_history,)
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.grid(True)

        plt.savefig(os.path.join(result_dir, 'learning_rate_plot-warm-5.png'))
        plt.close()


