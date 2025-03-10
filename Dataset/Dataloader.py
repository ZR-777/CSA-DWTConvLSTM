import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Any


class TEC_Dataset(Dataset):
    def __init__(self, path, train, time_step, predict_step):
        super(TEC_Dataset, self).__init__()
        self.path = path
        self.train = train
        self.time_step = time_step
        self.predict_step = predict_step

        self.high_data_path = os.path.join(self.path, "high_2013_2014_2015.npy")
        self.low_data_path = os.path.join(self.path, "low_2017_2018_2019.npy")
        self.high_data = (np.load(self.high_data_path))
        self.low_data = (np.load(self.low_data_path))
        final_2013_2014_2015_x, final_2013_2014_2015_y = self.split_data(self.high_data, self.time_step,
                                                                         self.predict_step)
        final_2017_2018_2019_x, final_2017_2018_2019_y = self.split_data(self.low_data, self.time_step,
                                                                         self.predict_step)
        if self.train:
            train_x = np.concatenate((final_2013_2014_2015_x[:-365], final_2017_2018_2019_x[:-365]), axis=0)
            train_y = np.concatenate((final_2013_2014_2015_y[:-365], final_2017_2018_2019_y[:-365]), axis=0)
            self.train_x = torch.from_numpy(train_x)
            self.train_y = torch.from_numpy(train_y)
            self.train_x = torch.unsqueeze(self.train_x, 2)
            self.train_y = torch.unsqueeze(self.train_y, 2)
            self.len = len(self.train_y)
            print("self.train_x.size():", self.train_x.size())
            print("self.train_y.size():", self.train_y.size())
        else:
            test_x = np.concatenate((final_2013_2014_2015_x[-365:], final_2017_2018_2019_x[-365:]), axis=0)
            test_y = np.concatenate((final_2013_2014_2015_y[-365:], final_2017_2018_2019_y[-365:]), axis=0)
            self.test_x = torch.from_numpy(test_x)
            self.test_y = torch.from_numpy(test_y)
            self.test_x = torch.unsqueeze(self.test_x, 2)
            self.test_y = torch.unsqueeze(self.test_y, 2)
            self.len = len(self.test_y)
            print("self.test_x.size():", self.test_x.size())
            print("self.test_y.size():", self.test_y.size())

    def __getitem__(self, index) -> Tuple[Any, Any]:
        if self.train:
            return self.train_x[index], self.train_y[index]
        else:
            return self.test_x[index], self.test_y[index]

    def __len__(self):
        return self.len

    def split_data(self, data, time_step, predict_step):
        x = []
        y = []
        for i in range(0, data.shape[0] - time_step - predict_step + 1, predict_step):
            x.append(data[i:i + time_step, :, :])
            y.append(data[i + time_step:i + time_step + predict_step, :, :])
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        return x, y


if __name__ == '__main__':
    train_data = TEC_Dataset(path='./', train=True, time_step=12, predict_step=12)
    train_dataloader = DataLoader(train_data, 10, pin_memory=True)
    for i, (inputs, targets) in enumerate(train_dataloader):
        print(inputs.size())
        print(targets.size())
