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
        self.cnt = 0

    def num_to_xy(self,num):
        x = int(num/8)
        y = num-x*8
        return x,y

    def change_turn(self):
        tmp = self.board[0].copy()
        self.board[0] = self.board[1]
        self.board[1] = tmp
        self.cnt += 1


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
    
    def possible_list(self):
        ret = np.array([],dtype='int8')
        for i in range(8):
            for j in range(8):
                if self.can_put(int(i*8+j)):
                    ret = np.append(ret,int(i*8+j))
        return ret
    
    def put(self,num):
        x,y = self.num_to_xy(num)
        arr = self.will_be_reversed(num)
        self.board[0] += arr
        self.board[1] -= arr
        self.board[0][x][y] = 1
    
    def you_cnt(self):
        return np.sum(self.board[0])
    
    def opo_cnt(self):
        return np.sum(self.board[1])
    
    def output_board(self):
        return self.board
    
    def print_board(self):
        print(" ",end="　")
        for i in range(1,9):
            print(i,end="　")
        print()

        for i in range(0,8):
            print("　────────────────────────")
            print(i+1,end="｜")
            for j in range(0,8):
                if self.board[0][i][j] == 1:
                    if self.cnt % 2 == 0:
                        print("○",end="｜")
                    else:
                        print("●",end="｜")
                elif self.board[1][i][j] == 1:
                    if self.cnt % 2 == 0:
                        print("●",end="｜")
                    else:
                        print("○",end="｜")
                else:
                    print(" ",end="｜")
            print()
        print("　────────────────────────")


# board = np.array([
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
# game = Game(board)
# game.print_board()

# game.put(4*8+4)
# game.change_turn()
# game.print_board()
