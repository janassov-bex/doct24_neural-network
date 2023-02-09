from typing import Union, Dict, List
from Models.ModelsAI.PredictLungMany import UNet_PredictLungMany
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import pickle

class ModelAI:
    def __init__(self, model, criterion, optimizer, path_model:str=None):
        self.model = model
        if path_model != None:
            try:
                print(self.model.load_state_dict(torch.load(path_model)))
            except:
                print("Модель не загрузилась")
        self.criteria = criterion
        self.optimazer = optimizer

class ModelsAI:
    def __init__(self):
        self.load_model_names()

    def load_model_names(self):
        unet_lung_many = UNet_PredictLungMany(in_channels=1, out_channels=5, device='cpu')
        self.model_names = {
            'LungMany': ModelAI(unet_lung_many,
                                nn.CrossEntropyLoss(),
                                optim.Adam(unet_lung_many.parameters(), lr=0.0001))
        }

        pass

    def predict_model(self, name_model:str, data_path:str, params=None):
        outputs = None
        if os.path.isfile(data_path):
            with open(data_path, 'rb') as file:
                inputs = file
            inputs = pickle.load(inputs)
            with torch.no_grad():
                self.model_names[name_model].optimizer.zero_grad()
                outputs = self.model_names[name_model].model(inputs)
        return outputs

    def get_model_names(self)->List[str]:
        return list(self.model_names.keys())
