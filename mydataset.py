#------------------------------------------------
#import module
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

#-----------------------------------------------
class myDataset(Dataset):
    def __init__(self, csv_path, target='Tavg', transform=None, numpy=False, unsq=False) -> None:
        #   target = 'Tavg', 'Tamp' or 'Trip
        df = pd.read_csv(csv_path)
        image_path  = df['path']
        labels      = df[target].astype('float32')
        
        self.image_path = image_path
        self.labels     = labels
        self.transform  = transform
        self.numpy      = numpy
        self.unsq       = unsq
    
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        input_path  = self.image_path[index]
        input_data  = Image.open(input_path)
        if self.transform is not None:
            input_data = self.transform(input_data)
        
        if self.numpy == True:
            input_data = torch.from_numpy(input_data)
            input_data = input_data.permute(2, 0, 1)
        
        if self.unsq == True:
            input_data = input_data.unsqueeze(0)
        
        img_path    = self.image_path[index]
        labels      = [self.labels[index]]
        labels      = torch.Tensor(labels)
        
        return input_data, labels, img_path