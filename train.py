import numpy as np
import matplotlib.pylab as plt # Matplotlibの中の一番使う部分
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader, random_split
import time
import math
import pprint

#変数
train = False
model_id = "1104_2048_192_7"
batch = 2048
ker_num = 192
ker_rep = 7

testloader_test = False
sample_test = True

start = time.time()

x = np.load("./data/x/all.npy")
y = np.load("./data/y/all.npy")

tensor_x = torch.Tensor(x)
tensor_y = torch.Tensor(y)
dataset = TensorDataset(tensor_x,tensor_y)

# データローダーの作成
data_len = len(dataset)
train_len = int(0.7 * data_len)
val_len = int(0.2 * data_len)
test_len = data_len - train_len - val_len
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

# ネットワークの定義
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 2*8*8
        self.conv = nn.Conv2d(2, 16, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16*8*8, 64)
        self.relu = nn.ReLU()
        # self.Lrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(128*8*8, 64)
        self.fc2 = nn.Linear(2*8*8, 2*8*8)
        self.fc3 = nn.Linear(2*8*8, 64)
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3,padding=1)

    def forward(self, x):
        for i in range(0,ker_rep):
            if i == 0:
                x = self.conv1(x)
            else :
                x = self.conv2(x)
            x = self.relu(x)
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        return x


# モデル、損失関数、最適化手法の設定
device = torch.device("cuda:0")
model = CNN()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.RMSprop(model.parameters(), lr=0.0002, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optimizer = optim.RAdam(model.parameters(), lr=0.001, betas=(0.99, 0.999), eps=1e-08, weight_decay=0)
# トレーニングループ
running_loss_list = []
val_loss_list = []
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')
        running_loss_list.append(running_loss / len(train_loader))
        val_loss_list.append(val_loss / len(val_loader))

# モデルのトレーニング
if(train):
    epoch = 50
    train_model(model, criterion, optimizer, train_loader, val_loader, epoch)
else:
    model = torch.load("./model/model_save_"+model_id)

# テストデータでの評価
model.eval()
test_loss = 0.0
correct = 0
total = 0

if(testloader_test):
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            __, answers = torch.max(labels,1)
            correct += (predicted == answers).sum().item()

    print("テスト損失:", round(test_loss / len(test_loader),5))
    print("テスト精度（正答率）:", str(round(correct / total * 100, 2)) +"%")

board_data = torch.Tensor([[
[[0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,1,0,0,0],
 [0,0,0,1,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0]],

[[0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,1,0,0,0,0],
 [0,0,0,0,1,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0]]]])

empty_board = [[0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0]]

# 打ち手を予想する
if(sample_test):
    with torch.no_grad():
        board_data = board_data.to(device)
        result = model(board_data)
        # print(result)
        res = torch.max(result,1)[1].item()
        empty_board[int(res/8)][res%8] = 1
        pprint.pprint(empty_board)

if(train):
    #グラフ描画用
    plt.figure(figsize=(6,6))
    plt.plot(range(epoch), running_loss_list)
    plt.plot(range(epoch), val_loss_list, c='#00ff00')
    plt.xlim(0, epoch)
    plt.ylim(0, 4.0)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['running_loss', 'val_loss'])
    plt.title('loss')
    plt.savefig("./loss/loss"+str(round(test_loss / len(test_loader)*1000))+".png")
    plt.clf()
    torch.save(model, './model/model_save_'+str(round(test_loss / len(test_loader)*1000)))

end = time.time()
s = math.floor((end-start)%60)
m = math.floor((end-start)/60)%60
h = math.floor((end-start)/3600)
print("所要時間: "+str(h)+"h "+str(m)+"m "+str(s)+"s")