import torch
import torch.nn as nn
import numpy as np
from rule import Game

# ネットワークの定義
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.fc1 = nn.Linear(128*8*8, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(0,7):
            if i == 0:
                x = self.conv1(x)
            else :
                x = self.conv2(x)
            x = self.relu(x)
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        return x
    
class Model():
    def __init__(self):        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model = torch.load("./model/model_save_1104_2048_192_7")
        self.model = self.model.to(self.device)

    def think(self,board): #board: np.array
        with torch.no_grad():
            board_data = torch.Tensor(np.array([board])).to(self.device)
            result = self.model(board_data)
            indices = torch.where(result > 0)
            result = result[torch.where(result > 0)]
            result = torch.sort(result, descending=True)
            indices = indices[1][result.indices]

            game = Game(board)
            for i in range(len(indices)):
                if game.can_put(int(indices[i])):
                    return int(indices[i])
            
            return -1

# board_data = np.array([
#     [[0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,1,0,0,0],
#     [0,0,0,0,1,1,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0]],

#     [[0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,1,0,0,0,0],
#     [0,0,0,1,0,0,0,0],
#     [0,0,0,1,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0]]])

# model = Model()
# print(model.think(board_data))