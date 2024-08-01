import numpy as np
from rule import Game
from nn import Model,CNN
import random

board_black = np.zeros(shape=(8,8), dtype='int8')
board_black[3][4] = 1
board_black[4][3] = 1
board_white = np.zeros(shape=(8,8), dtype='int8')
board_white[3][3] = 1
board_white[4][4] = 1
board = np.array([board_black,board_white])
game = Game(board)
model = Model()

turn = -1 # 1:プレイヤー -1:コンピューター
passed = False #前回がパスか

while(1):
    game.print_board()
    if turn > 0:
        list = game.possible_list()
        if len(list) > 0:
            print("あなたの番です")
            while(1):
                try:
                    x,y = (int(x)-1 for x in input().split())
                    num = x*8+y
                    if game.can_put(num):
                        game.put(num)
                        passed = False
                        break
                    else:
                        print("そこには打てません")
                except:
                    print("そこには打てません")
        else:
            print("打てる場所がありません。Enterでパスしてください")
            input()
            if(passed):
                you = game.you_cnt()
                opo = game.opo_cnt()
                print("あなた"+str(you)+"石 "+"相手"+str(opo)+"石")
                if you>opo:
                    print("あなたの勝ち")
                elif you<opo:
                    print("あなたの負け")
                else:
                    print("引き分け")
                break
            else:
                passed = True
    else:
        print("コンピューターの番です")
        think_num = model.think(game.output_board())
        if think_num >= 0:
            game.put(think_num)
            print(int(think_num/8)+1,(think_num%8)+1)
            passed = False
        else:
            list = game.possible_list()
            if len(list) > 0:
                #TODO
                print("コンピューター困惑中...")
                num = random.choice(list)
                game.put(num)
                print(int(num/8)+1,(num%8)+1)
                passed = False
            else:
                print("打てる場所がありません。パスします")
                if(passed):
                    opo = game.you_cnt()
                    you = game.opo_cnt()
                    print("あなた"+str(you)+"石 "+"相手"+str(opo)+"石")
                    if you>opo:
                        print("あなたの勝ち")
                    elif you<opo:
                        print("あなたの負け")
                    else:
                        print("引き分け")
                    break
                else:
                    passed = True

    game.change_turn()
    turn*=-1

        
