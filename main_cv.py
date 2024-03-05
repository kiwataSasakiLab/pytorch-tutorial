#------------------------------------------------
#import module
import time
import mydataset
import torch
import torchvision
import numpy as np
import random
from torch                      import nn
from sklearn.model_selection    import KFold
from torch.utils.data           import DataLoader, SubsetRandomSampler
from torchvision                import transforms
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#------------------------------------------------
#   Methods
#   (1)     Training method
#   (2)     Test method
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.sampler)
    model.train()
    train_loss = 0
    num_train = 0
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
    print("Train Loss:", train_loss)

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    num_test = 0
    pred_list = []
    label_list = []
    with torch.no_grad():
        for data in dataloader:
            num_test += 1
            
            input_data, target = data[0].to(device), data[1].to(device)
            pred = model(input_data)
            test_loss += loss_fn(pred, target).item()

            for tensor in pred:
                pred_list.append([tensor.item()])
            for tensor in target:
                label_list.append([tensor.item()])
    
    test_loss /= num_test
    print("Test Loss:", test_loss)
    
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

Torque      = 'Tavg'                                #   Tavg or Tpp
csv_path    = './dataloader/Material_Torque.csv'    #   You must make this csv file.

transform = transforms.Compose([
    transforms.ToTensor()
])
CV_dataset          = mydataset.myDataset(csv_path=csv_path, target=Torque, transform=transform)
train_test_split    = KFold(n_splits=KFOLD_NUM, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(train_test_split.split(np.arange(len(CV_dataset)))):
    train_SubSampler    = SubsetRandomSampler(train_idx)
    test_SubSampler     = SubsetRandomSampler(test_idx)
    train_dataloader    = DataLoader(CV_dataset, BATCH_SIZE, sampler=train_SubSampler,  num_workers=1, pin_memory=True)
    test_dataloader     = DataLoader(CV_dataset, BATCH_SIZE, sampler=test_SubSampler,   num_workers=1, pin_memory=True)
    
    print('train data size', len(train_dataloader.sampler))
    print('test data size', len(test_dataloader.sampler))
    model = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)
    for layer, param in enumerate(model.features.parameters()):
        param.requires_grad = False
        if(layer == 4*TRANS_LAYER-1):
            break
    model.classifier = FC_layer
    model = model.to(device)

    #   Loss Function
    loss_fn = nn.MSELoss()

    #   Optimizer
    optimizer       = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    #   Learning and Validation
    for epoch in range(EPOCH):
        print(f"Fold {fold+1} : Epoch {epoch+1}\n--------------------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader,   model, loss_fn, fold+1)

t_end = time.time()
print(t_end - t_start , ' [sec]')