import torch
from torch import nn
import os, time
import numpy as np
from Network import ViT
from ImageUtils import parse_record
from DataLoader import custom_dataloader
import timeit
import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = ViT(configs)

    def model_setup(self, configs):

        random_seed = configs.random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr = configs.learning_rate, 
                                          betas = configs.betas, 
                                          weight_decay = configs.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                              milestones = configs.scheduler_milestones, 
                                                              momentum = configs.momentum)
        
    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):

        self.model_setup(configs)

        start = timeit.default_timer()

        train_dataloader = custom_dataloader(x_train, y_train, batch_size=configs.batch_size, train=True)
        val_dataloader = custom_dataloader(x_valid, y_valid, batch_size=configs.batch_size, train=True)

        for epoch in tqdm(range(configs.num_epochs), position=0, leave=True):

            self.network.train()

            train_labels = []
            train_preds = []
            train_running_loss = 0

            for idx, (images, labels) in enumerate(tqdm(train_dataloader, position=0, leave=True)):

                processed_images =[]
                
                for image in images:
                    processed_images.append(parse_record(image, True))

                current_images = torch.tensor(np.array(processed_images)).float().to('cuda')
                current_labels = torch.tensor(labels).to('cuda')

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
                for idx, (images, labels) in enumerate(tqdm(val_dataloader, position=0, leave=True)):

                    val_processed_images = []
                    
                    for image in images:
                        val_processed_images.append(parse_record(image, True))

                    validation_images = torch.tensor(np.array(val_processed_images)).float().to('cuda')
                    validation_labels = torch.tensor(labels).to('cuda')

                    val_predictions = self.network(validation_images)
                    val_prediction_labels = torch.argmax(val_predictions, dim=1)

                    val_labels.extend(validation_labels.cpu().detach())
                    val_preds.extend(val_prediction_labels.cpu().detach())

                    loss = self.cross_entropy_loss(val_prediction_labels, validation_labels)
                    val_running_loss += loss.item()
            val_loss = val_running_loss / (idx + 1)

            print("-"*30)
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print(f"Train Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
            print(f"Valid Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
            print("-"*30)

        if epoch % configs.save_interval == 0:
            self.save(epoch)

        stop = timeit.default_timer
        print(f"Training Time: {stop - start : .2f}s")

    def evaluate(self, x, y):

        test_dataloader = custom_dataloader(x, y, train=False)

        self.network.eval()
            
        test_labels_final = []
        test_preds_final = []

        with torch.no_grad():
            for idx, (images, labels) in enumerate(tqdm(test_dataloader, position=0, leave=True)):

                test_processed_images = []
                
                for image in images:
                    test_processed_images.append(parse_record(image, False))

                test_images = torch.tensor(np.array(test_processed_images)).float().to('cuda')
                test_labels = torch.tensor(labels).to('cuda')

                test_predictions = self.network(test_images)
                test_prediction_labels = torch.argmax(test_predictions, dim=1)

                test_labels_final.extend(test_labels.cpu().detach())
                test_preds_final.extend(test_prediction_labels.cpu().detach())

            y_preds = torch.tensor(test_prediction_labels)
            y_labels = torch.tensor(y)

        print(f"Test Accuracy: {torch.sum(y_preds==y_labels)/y_labels.shape[0]:.4f}")

    def predict_prob(self, x):
        pass

    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs.save_dir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.configs.model_dir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))