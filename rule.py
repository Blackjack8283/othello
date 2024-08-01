import numpy as np

class Game:
    def __init__(self,board):
        # board_black = np.zeros(shape=(8,8), dtype='int8')
        # board_black[3][4] = 1
        # board_black[4][3] = 1
        # board_white = np.zeros(shape=(8,8), dtype='int8')
        # board_white[3][3] = 1
        # board_white[4][4] = 1
        # self.board = np.array([board_black,board_white])
        self.board = board

    def num_to_xy(self,num):
        x = int(num/8)
        y = num-x*8
        return x,y

    def change_turn(self):
        tmp = self.board[0].copy()
        self.board[0] = self.board[1]
        self.board[1] = tmp

    def will_be_reversed(self,num): #ひっくり返せる場所を返す
        x,y = self.num_to_xy(num)
        ret = np.zeros(shape=(8,8), dtype='int8')
        dx = np.array([1,0,-1,0,1,1,-1,-1])
        dy = np.array([0,-1,0,1,1,-1,1,-1])

        if self.board[0][x][y]!=0 or self.board[1][x][y]!=0:
            return ret
        
        for i in range(8):
            cnt = 0
            nx=x; ny=y
            while(1):
                nx += dx[i]; ny += dy[i]

                if (nx<0 or ny<0 or nx>=8 or ny >= 8) or (self.board[0][nx][ny]==0 and self.board[1][nx][ny]==0):
                    break
                elif self.board[0][nx][ny]==1:
                    for k in range(1,cnt+1):
                        nnx = x+dx[i]*k; nny=y+dy[i]*k
                        ret[nnx][nny] = 1
                    break
                else:
                    cnt+=1
        return ret
    
    def can_put(self,num):
        arr = self.will_be_reversed(num)
        if arr.sum()!=0:
            return True
        else:
            return False
    
    def possible_list(self,num):
        ret = np.zeros(shape=(8,8), dtype='int8')
        for i in range(8):
            for j in range(8):
                if self.can_put(num):
                    ret[i][j] = 1
        return ret
    
    def put(self,num):
        x,y = self.num_to_xy(num)
        arr = self.will_be_reversed(num)
        self.board[0] += arr
        self.board[1] -= arr
        self.board[0][x][y] = 1

# othello = Game()

# othello.board = np.array([
# [[0,0,0,0,0,0,0,0],
#  [0,1,1,1,0,1,1,1],
#  [0,1,0,0,0,0,0,1],
#  [0,1,0,0,0,0,0,1],
#  [0,1,0,0,0,0,0,0],
#  [0,1,0,0,0,0,0,1],
#  [0,1,0,0,0,0,0,1],
#  [0,1,1,1,1,1,1,1]],

# [[0,0,0,0,1,0,0,0],
#  [0,0,0,0,1,0,0,0],
#  [0,0,1,1,1,1,1,0],
#  [0,0,1,1,1,1,1,0],
#  [0,0,1,1,0,1,1,1],
#  [0,0,1,1,1,1,1,0],
#  [0,0,1,1,1,1,1,0],
#  [0,0,0,0,0,0,0,0]]])

# # othello.put(4,4)
# # othello.change_turn()
# print(othello.possible_list())