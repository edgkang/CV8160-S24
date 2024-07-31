import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np


class TrafficDataset(Dataset):
    def __init__(self, file_path: str, window: int, horizon: int, task: str='train', val_size: float=0.2):
        super().__init__()
        assert task=='train' or task=='validation', \
            print("Choose either train or validation")
        df = pd.read_pickle(file_path)
        self.window = window
        self.horizon = horizon
        self.tps_df = self.reshape_df(df)
        #fill the nans with zero
        self.tps_df = self.tps_df.fillna(0)
        self.X = []
        self.y = []
        self.setup_forecast()

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=val_size, random_state=1)
        if task=='train':
            self.inputs = X_train
            self.targets = y_train
        if task=='validation':
            self.inputs = X_val
            self.targets = y_val

    def setup_forecast(self):
        rows, cols = self.tps_df.shape
        df_time = pd.DataFrame(columns=["time", "dow", "hour"])
        df_time["time"] = pd.to_datetime(self.tps_df.index)
        df_time["dow"] = df_time["time"].dt.day_of_week
        df_time["hour"] = df_time["time"].dt.hour
        for col in range(cols):
            col_vals = self.tps_df.iloc[:, col].values
            for t in range(0,(rows-(self.window+self.horizon))):
                #takes into account the traffic index, day of week and time of day
                x = np.hstack((col_vals[t:t+self.window].reshape(self.window, 1), df_time.iloc[t:t+self.window, 1].values.reshape(self.window, 1)
                               , df_time.iloc[t:t+self.window, 2].values.reshape(self.window, 1)))
                # x = col_vals[t:t+self.window]
                y = col_vals[t+self.window:(t+self.window+self.horizon)]
                self.X.append(x)
                self.y.append(y)

    def reshape_df(self, tps_df):
        reshaped_tps_df = pd.DataFrame()
        reshaped_tps_df['TIME'] = tps_df.time.unique()
        for seg in tps_df.segmentID.unique():
            column = tps_df[tps_df['segmentID'] == seg][['time','TrafficIndex_GP']].drop_duplicates(subset=['time'])
            column.columns = ['TIME', str(seg)]
            reshaped_tps_df = reshaped_tps_df.join(column.set_index('TIME'), on='TIME')

        reshaped_tps_df = reshaped_tps_df.set_index('TIME')
        return reshaped_tps_df
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        X = torch.tensor(self.inputs[index],dtype=torch.float32) #.reshape(self.window,3)
        y = torch.tensor(self.targets[index],dtype=torch.float32)

        return {'inputs':X,'outputs':y}
    
