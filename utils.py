import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

def training_model(model, criterion, optimizer, train_loader, val_loader, n_epochs, device):
    mean_train_loss = []
    train_losses = []
    val_losses = []
    mean_val_loss = []
    # ts = ToTensor()
    for it in range(n_epochs):
    # zero the parameter gradients
        for i_batch, sample_batched in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            outputs = model(sample_batched['inputs'].to(device))
            loss = criterion(outputs, sample_batched['outputs'].to(device))
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        for i_val, sample_val in enumerate(val_loader):
            outputs = model(sample_val['inputs'].to(device))
            val_loss = criterion(outputs, sample_val['outputs'].to(device))
            val_losses.append(val_loss.item())

        mean_train_loss.append(np.mean(train_losses))
        mean_val_loss.append(np.mean(val_losses))
        if (it+1) % 1 == 0:
            print(f'Epoch {it+1}/{n_epochs}, Training Loss: {np.mean(train_losses):.4f}, Validation Loss: {np.mean(val_losses):.4f}')
    
    return mean_train_loss, mean_val_loss, train_losses, val_losses, model


def evaluataion(model, model_path, validation_loader, device):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    rmse = []
    maes = []

    for i_val,sample_val in enumerate(validation_loader):
        preds = model(sample_val['inputs'].to(device))
        targets = sample_val['outputs'].to(device)
        error = ((preds-targets)**2).sum(1).mean().detach().cpu().numpy()
        mae = (preds-targets).abs().sum(1).mean().detach().cpu().numpy()
        rmse.append(math.sqrt(error))
        maes.append(mae)

    return {'rmse':np.mean(rmse),'mae':np.mean(maes)}

class TestSetup():
    def __init__(self, model, model_path, test_filepath, device, window, horizon):

        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.device = device
        df = pd.read_pickle(test_filepath)
        self.tps_df = self.reshape_df(df)
        self.df_time = pd.DataFrame(columns=["time", "dow", "hour"])
        self.df_time["time"] = pd.to_datetime(self.tps_df.index)
        self.df_time["dow"] = self.df_time["time"].dt.day_of_week
        self.df_time["hour"] = self.df_time["time"].dt.hour
        self.horizon = horizon
        self.window = window
        self.pred = None

    def reshape_df(self, tps_df):
        reshaped_tps_df = pd.DataFrame()
        reshaped_tps_df['TIME'] = tps_df.time.unique()
        for seg in tps_df.segmentID.unique():
            column = tps_df[tps_df['segmentID'] == seg][['time','TrafficIndex_GP']].drop_duplicates(subset=['time'])
            column.columns = ['TIME', str(seg)]
            reshaped_tps_df = reshaped_tps_df.join(column.set_index('TIME'), on='TIME')

        reshaped_tps_df = reshaped_tps_df.set_index('TIME')
        return reshaped_tps_df
    
    def prediction(self):
        self.pred = pd.DataFrame()
        rows, cols = self.tps_df.shape
        time_range = pd.date_range(start=self.tps_df.index[-1], freq='15min', periods=13)
        self.pred.index = time_range[1:]

        for col in range(cols): #iterate through the columns and get a prediction
            input = np.hstack(
                (
                    self.tps_df.iloc[(rows-self.window): , col].values.reshape(self.window, 1),
                    self.df_time.iloc[(rows-self.window):, 1].values.reshape(self.window, 1),
                    self.df_time.iloc[(rows-self.window):, 2].values.reshape(self.window, 1)
                )
            )
            input = torch.tensor(input, dtype=torch.float).unsqueeze(0)
            preds = self.model(input.to(self.device))
            self.pred[self.tps_df.columns[col]] = np.round(preds.detach().cpu().numpy().squeeze(), decimals=3)
            #extend the tps_df column to add the the prediction
    
    def save_results(self, file_name):
        if self.pred is None:
            self.prediction()
        self.pred.to_json(file_name)
