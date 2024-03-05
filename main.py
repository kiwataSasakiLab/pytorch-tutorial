#------------------------------------------------
#import module
import time
import csv
import mydataset
import torch
import torchvision
import numpy as np
import random
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#------------------------------------------------
#   Methods
#   (1)     Training method
#   (2)     Test method
def train(dataloader, model, loss_fn, optimizer, history):
    size = len(dataloader.sampler)
    model.train()
    train_loss  = 0
    num_train   = 0
    for batch, data in enumerate(dataloader):
        num_train += 1
        input_data, target = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        pred = model(input_data)
        loss = loss_fn(pred, target)
        train_loss += loss.item()
        
        #   Backprobagation
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(input_data)
            print(f"loss: {loss:>10f} [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_train
    history['train_loss'].append(train_loss)
    print("Train Loss:", train_loss)

def test(dataloader, model, loss_fn, history, epoch):
    model.eval()
    test_loss   = 0
    num_test    = 0
    pred_list   = []
    label_list  = []
    path_list   = []
    with torch.no_grad():
        for data in dataloader:
            num_test += 1
            input_data, target, img_path = data[0].to(device), data[1].to(device), data[2]
            pred = model(input_data)
            test_loss += loss_fn(pred, target).item()
            if epoch+1 == EPOCH:
                for tensor in pred:
                    pred_list.append([tensor.item()])
                for tensor in target:
                    label_list.append([tensor.item()])
                for img in img_path:
                    path_list.append(img)
    test_loss /= num_test
    history['test_loss'].append(test_loss)
    
    with open('cnn.csv', 'w', newline="") as f:
        wrt = csv.writer(f)
        wrt.writerows(pred_list)
    with open('fem.csv', 'w', newline="") as f:
        wrt = csv.writer(f)
        wrt.writerows(label_list)
    
    path_list = '\n'.join(path_list)
    with open('path_list.csv', 'w') as f:
        f.write(path_list)
    
    print("Test Loss:", test_loss)

def SaveLoss(history):
    train_loss = np.array(history['train_loss'])
    test_loss = np.array(history['test_loss'])
    np.savetxt('train_loss.csv', train_loss)
    np.savetxt('test_loss.csv', test_loss)
    
#------------------------------------------------
#   Hyper Parameters
EPOCH           = 50
BATCH_SIZE      = 64
NUM_CLASS       = 1
TRANS_LAYER     = 7                                 #   1 ~ 13, if you use VGG16
KFOLD_NUM       = 5
LEARNING_RATE   = 1e-4

#   Fully connected layer for VGG16
FC_layer = nn.Sequential(
    nn.Linear(in_features=25088, out_features=256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=256, out_features=NUM_CLASS)
)

#------------------------------------------------
t_start = time.time()

device  = "cuda" if torch.cuda.is_available() else "cpu"

csv_path    = 'Material_Torque.csv' # dataset csv file
target_data = 'Tavg'                # Tavg or Tpp
weight_name = 'Model_Tavg.pth'

transform = transforms.Compose([
    transforms.ToTensor()
])
full_dataset = mydataset.myDataset(csv_path=csv_path, target=target_data, transform=transform)

train_size  = int(0.8 * len(full_dataset))
test_size   = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(
    full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
)
train_dataloader    = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
test_dataloader     = DataLoader(test_dataset,  BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
print('train data : ', len(train_dataloader.dataset))
print('test data  : ', len(test_dataloader.dataset))

model = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)
for layer, param in enumerate(model.features.parameters()):
    param.requires_grad = False
    if(layer == 4*TRANS_LAYER-1):
        break
model.classifier = FC_layer
model = model.to(device)
print(model)

loss_fn         = nn.MSELoss()
optimizer       = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
history         = {'train_loss':[], 'test_loss':[]}

for epo in range(EPOCH):
    print(f"Epoch {epo+1}\n--------------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer,  history)
    test(test_dataloader,   model, loss_fn, history,    epo)
torch.save(model.state_dict(), weight_name)
SaveLoss(history)
        
t_end = time.time()
print(t_end - t_start , ' [sec]')