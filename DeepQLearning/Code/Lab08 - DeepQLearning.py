# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import collections
import enum
from shutil import move
import numpy as np
from PIL import Image,ImageTk
import random
import time
from requests import head
import torch
from ML import Snake_learning 
try:
    import tkinter as tk
    import tkinter.simpledialog as simpledialog
except ImportError:
    try:
        import Tkinter as tk
    except ImportError:
        print("Missing library: Tkinter, please install")
#https://github.com/vedantgoswami/SnakeGameAI/blob/main/agent.py

# NN VARIABLES
MEMORY_CAPACITY = 20 # Es un escenario pequeño deberia de aprender rapido 
BATCH_SIZE = 10 # Caso similar al anterior
CITER=5
LEARNING_RATE = 0.001# Esto puede ser [0.1,0.01,0.001]
DISCOUNT = 0.9  # Esto puede ser [0.1,0.01,0.001]
GREDDY = 0.50 # Puede subirle mas, pero la idea es que una vez es aleaotorio y otras no
MINGREDDY = 0.80 # Por que no quiere moverse
DECAY = 0.00001 # Por que no quiere moverse (si lo aumenta seguiria subiendo greedy, tenga cuidado)

# serian los siguientes hiperparametros 
# REDES = leaky,tan,relu
# LR = [0.1 - 0.001] Luego puede probar en el extremo
# GREDDY = [0.1 - 0.001] Puede probar en el extremo
# Neuronas = [50,1000]

# RANDOM SEEDS
RANDOM_SEED = 21
SNAKE_SEED = 31337

# ENVIROMENT VARIABLES
WIDTH = 16
HEIGHT = 24
INIT_LENGTH = 3
INIT_MOVES = 256
INIT_DIRECTION = 0
EXTRA_MOVES = 128

#REWARDS
EMPTY_REWARD = -1
APPLE_REWARD = 100
WALL_REWARD = -100
SELF_REWARD = -100

# Pensamientos , era mejor que vayan directo al grano

### UI related
FPS = 30
APS = 8
# Colors
colors = {
    0: (0,0,0),
    1: (255,255,128),
    2: (0,255,0),
    3: (255,0,0)
}

# Agent Constants 
FRONT_MOVE_HORIZONTAL = {0:0,1:1,2:0,3:-1}
FRONT_MOVE_VERTICAL = {0:-1,1:0,2:1,3:0}
LEFT_MOVE_HORIZONTAL ={0:-1,1:0,2:1,3:0}
LEFT_MOVE_VERTICAL = {0:0,1:-1,2:0,3:1}
RIGHT_MOVE_HORIZONTAL = {0:1,1:0,2:-1,3:0}
RIGHT_MOVE_VERTICAL = {0:0,1:1,2:0,3:-1}
PROXIMITY_SENSOR_LENGHT = 3

### Q LEARNING VARIABLES ###
AC = "action"
DF = "DiscountFactor"
EP = "EpsilonGreddy"
DE = "Decay" 
### NN VARIABLES###
MA = "Memory Actions"
MS = "Memory States"
MR = "Memory Rewards"
BS = "Batch Size"
I = "C Iterations"


class Action(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Agent():
    # Initializes the training model
    # Input states for the model depend on the get_state method, which can be modified


    def __init__(self, memory_capacity, batch_size, c_iters, learning_rate, discount_factor, eps_greedy, decay):
        # Initialize the two main vectors
        self.STATS =  dict()
        self.NN = dict()
        # The actions that I take depends on my current direction, the snake only chose
        # (0 = Continue direction, 1 = Right, 2 = Left)
        # (Current movement, NN decision(Snake Decision))
        self.STATS[AC] = {(0,0):0,(0,1):1,(0,2):3,
                        (1,0):1,(1,1):2,(1,2):0,
                        (2,0):2,(2,1):3,(2,2):1,
                        (3,0):3,(3,1):0,(3,2):2} 
        self.STATS[EP] = eps_greedy  
        self.STATS[DE] = decay
        self.prng = random.Random()
        self.prng.seed(RANDOM_SEED)
        self.brain = Snake_learning(learning_rate,memory_capacity,batch_size,c_iters,discount_factor)
        self.current_step = 0
        self.total_steps = 0
        self.total_simulation = 0
        self.best_apples = 0
        
    # Performs a complete simulation by the agent, then samples "batch" memories from memory and learns, afterwards updates the decay
    def simulation(self, env):
        self.current_step=0
         # Reset enviroment to begin a new simulation
        env.reset()
        # Changes made by decay just affect eps_greddy
        eps_greddy = self.STATS[EP]
        # Do simulation
        simulation_flag = False
        current_apples = 0
        while not simulation_flag:
            # Do the step
            current_apples += self.step(env) # step_tuple = ((reward, state),action)  
            # Change the new eps_greddy
            self.STATS[EP] *= (1+self.STATS[DE])
            # Ask for the end game
            simulation_flag = env.is_terminal_state() 
        self.total_simulation +=1
        self.STATS[EP] = eps_greddy
        # if(current_apples>2):
        #     print("manzana: ", current_apples)
        #     print("Paso en simulacion: ",self.current_step)
        self.total_steps+=self.current_step
        print("Simulacion: ",self.total_simulation)
        print("Pasos tomados: ",self.total_steps)
        if(self.total_steps > 20):
            self.simulation_apples(env)

    def simulation_apples(self, env):
        env.reset()
        simulation_flag = False
        current_apples = 0
        movements = 0
        while not simulation_flag:
            # Do the step
            current_apples += self.step(env,False) # step_tuple = ((reward, state),action)  
            # Ask for the end game
            movements +=1
            simulation_flag = env.is_terminal_state() 
            if movements > 30:
                simulation_flag = True
        if(current_apples>self.best_apples):
            self.best_apples = current_apples
            self.brain.save_model_perfect(current_apples)
            print("Mejor Score: ", current_apples)

    
    # Performs a single step of the simulation by the agent, if learn=False memories are not stored
    def step(self, env, learn=True):
        self.current_step+=1
        # Get current state
        state = env.get_state() # It hasn´t been tested
        random_movement = False
        movement = 0
        apple = 0
        # if learn=False no updates are performed and the best action is always taken
        if learn:
            eps_greddy = self.prng.random()
            if(self.STATS[EP]> MINGREDDY):
                self.STATS[EP] =MINGREDDY

            if eps_greddy > self.STATS[EP]:
                random_movement = True
        # if random action happen, choose random action
        if random_movement:
            movement = self.prng.randint(0,2) 
        else:
            movement = self.brain.best_movement(state[0])
        # Perform the action selected
        action =  self.STATS[AC][(state[1],movement)]
        env_tuple = env.perform_action(action)
        reward = env_tuple[0]
        if(reward == APPLE_REWARD):
            apple = 1
        if learn:
            if env.is_terminal_state():
                next_state = None
            else:
                next_state = torch.tensor([env_tuple[1][0]],dtype = torch.float32)
            reward = torch.tensor([reward])
            action =  torch.tensor([movement])
            state = torch.tensor([state[0]],dtype = torch.float32) # voy
            if(self.current_step % 50 == 0):
                print(self.current_step)
            self.brain.new_memory(state[0], action, next_state, reward)
            self.brain.train(self.prng)
        return apple





class Snake():
    def __init__(self, seed=SNAKE_SEED):
        self.prng = random.Random()
        self.seed = seed
        self.reset()
    
    def get_score(self):
        return self.score

    # Resets the environment
    def reset(self):
        self.prng.seed(self.seed)
        self.board = np.zeros((HEIGHT, WIDTH))
        self.valid_positions = set((i,j) for i in range(HEIGHT) for j in range(WIDTH))
        self.positions = collections.deque([])
        for i in range(INIT_LENGTH):
            self.positions.append(((HEIGHT//2)+i, WIDTH//2))
            self.board[self.positions[-1]] = 1 + int(i==0)
        self.direction = 0 # 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
        self.moves_left = INIT_MOVES
        self.dead = False
        self.score = 0
        self.__add_apple()
    
    def __add_apple(self):
        options = list(self.valid_positions - set(self.positions))
        if not options: return False
        apple = self.prng.choice(options)
        self.board[apple] = 3
        self.apple_position = apple
        return True

    # Returns action list            
    def get_actions(self):
        return [a.value for a in Action]

    # Get dangers arround snake
    def danger_inspection(self,snake_body,movement):
        # Wall inspection
        if movement[0]<0 or movement[0]>= HEIGHT or movement[1] <0 or movement[1]>= WIDTH:
            return -1
        # Body inspection
        if torch.any(torch.logical_and(torch.isin(snake_body[0],movement[0]),torch.isin(snake_body[1],movement[1]))).item():
            return -1
        return 1

    # Get rewards arround snake
    def reward_inspection(self,apple,movement):
        if movement[0] == apple[0] and movement[1]== apple[1]:
            return 1
        return 0

    # Get near_reward 
    def near_reward(self,apple,head,movement):
        reward = 0
        if head[0] == apple[0] or head[1]== apple[1]:
            # Near in vertical position
            if apple[0] == head[0]:
                if apple[1] == (head[1] + movement[1]):
                    reward = 1
            else:
                if apple[0] == (head[0] + movement[0]):
                    reward = 1
        return reward



    # Get reward proximity
    def sensor_proximity(self,apple,movement,body,head):
        pivot = 0
        while True:
            if self.reward_inspection(apple,(head[0]+movement[0]*pivot,head[1]+movement[1]*pivot)) == 1:
                pivot = 1
                break
            pivot+=1
            if self.danger_inspection(body,(head[0]+movement[0]*pivot,head[1]+movement[1]*pivot)) == -1:
                pivot = -1
                break
            
        return pivot
        # for i in range(1,PROXIMITY_SENSOR_LENGHT+1):
        #     if self.danger_inspection(body,(head[0]+movement[0]*i,head[0]+movement[0]*i)) == 1:
        #         return 1
        #     if self.reward_inspection(apple,(head[0]+movement[0]*i,head[0]+movement[0]*i)) == 1:
        #         return 2
        # return 0

    # To get a better understanding inwo own state
    def change_state(self, state):
        new_state = []
        # Get important positions
        body = torch.tensor(np.array(np.where( state[0] == 1)))
        head = torch.tensor(torch.where( state[1] == 1))
        apple = torch.tensor(torch.where( state[2] == 1))
        
        # Get current snake direction
        direction = self.direction
        # Get possible new movements
        front_movement = (FRONT_MOVE_VERTICAL[direction],FRONT_MOVE_HORIZONTAL[direction])
        right_movement = (RIGHT_MOVE_VERTICAL[direction],RIGHT_MOVE_HORIZONTAL[direction])
        left_movement = (LEFT_MOVE_VERTICAL[direction],LEFT_MOVE_HORIZONTAL[direction])
        front_decision = (head[0] + front_movement[0],head[1] + front_movement[1])
        right_decision = (head[0] + right_movement[0],head[1] + right_movement[1])
        left_decision = (head[0] + left_movement[0],head[1] + left_movement[1])
        # Get details in every possible movement
        # Reward Proximity
        new_state.append(self.sensor_proximity(apple,front_movement,body,head))
        new_state.append(self.sensor_proximity(apple,right_movement,body,head))
        new_state.append(self.sensor_proximity(apple,left_movement,body,head))
        # Dangers 
        new_state.append(self.danger_inspection(body,front_decision))
        new_state.append(self.danger_inspection(body,right_decision))         
        new_state.append(self.danger_inspection(body,left_decision ))  
        # Rewards
        new_state.append(self.reward_inspection(apple,front_decision))
        new_state.append(self.reward_inspection(apple,right_decision))       
        new_state.append(self.reward_inspection(apple,left_decision ))

        return new_state
    ### TODO(OPTIONAL): CAN BE MODIFIED; CURRENTLY RETURNS TENSOR OF SHAPE (3, WIDTH, HEIGHT)



    # Returns a sensors in snake and its own current direction
    def get_state(self):
        snake = torch.zeros((HEIGHT,WIDTH),dtype=torch.float32)
        head = torch.zeros((HEIGHT,WIDTH),dtype=torch.float32)
        apple = torch.zeros((HEIGHT,WIDTH),dtype=torch.float32)
        for pos in self.positions:
            snake[pos] = 1
        head[self.positions[0]] = 1 # Modificado por Juan
        apple[self.apple_position] = 1
        sensors = self.change_state(torch.stack((snake, head, apple)))
        return (sensors,self.direction)
    
    # Returns whether current state is terminal
    def is_terminal_state(self):
        return self.dead or self.moves_left<=0
    
    # Performs an action and returns its reward and the new state
    def perform_action(self, action):
        reward = EMPTY_REWARD
        if abs(self.direction - action)!=2: self.direction = action
        tail = self.positions.pop()
        self.board[tail] = 0
        if self.direction%2 == 0:
            head = (self.positions[0][0] + (-1 if self.direction==0 else 1), self.positions[0][1])
            if head[0]<0 or head[0]>=HEIGHT:
                self.dead = True
                reward = WALL_REWARD
        else:
            head = (self.positions[0][0], self.positions[0][1] + (-1 if self.direction==3 else 1))
            if head[1]<0 or head[1]>=WIDTH:
                self.dead = True
                reward = WALL_REWARD
        if not self.dead:
            self.board[self.positions[0]] = 1
            self.positions.appendleft(head)
            value = self.board[head]
            self.board[head] = 2
            if value<3 and value>0:
                self.dead = True
                reward = SELF_REWARD
            if value==3:
                self.board[tail] = 1
                self.positions.append(tail)
                if not self.__add_apple(): self.dead = True
                self.score += 1
                reward = APPLE_REWARD
        return reward,self.get_state()

class mainWindow():
    def __init__(self,agentCls=Agent):
        self.snake = Snake()
        self.agentCls = agentCls
        # Control
        self.redraw = False
        self.playing_user = False
        self.playing_agent = False
        self.last_direction = 0
        self.last_action = 0
        self.epoch = 0
        self.score = 0
        self.highscore = 0
        

        self.memory_capacity = MEMORY_CAPACITY#10000
        self.batch_size = BATCH_SIZE#1000
        self.c_iters = CITER#10
        self.learning_rate = LEARNING_RATE#0.001
        self.discount = DISCOUNT#0.25
        self.greedy = GREDDY#0.25
        self.decay = DECAY#1e-7
        self.agent = agentCls(self.memory_capacity, self.batch_size, self.c_iters, self.learning_rate, self.discount, self.greedy, self.decay)
        # Interface
        self.root = tk.Tk()
        self.root.title("Snake AI")
        self.root.bind("<Configure>",self.resizing_event)
        self.root.bind("<Key>",self.key_press_event)
        self.frame = tk.Frame(self.root, width=650, height=550)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=1,height=1)
        # Control buttons
        self.buttonReset = tk.Button(self.frame, text="Reset",command=self.buttonReset_press,bg="indian red")
        self.buttonStep = tk.Button(self.frame, text="Step",command=self.buttonStep_press,bg="sea green")
        self.buttonSkip = tk.Button(self.frame, text="Skip",command=self.buttonSkip_press,bg="sea green")
        self.buttonPlayAgent = tk.Button(self.frame, text="Simulation",command=self.buttonPlayagent_press,bg="forest green")
        self.buttonPlayUser = tk.Button(self.frame, text="Play",command=self.buttonPlayuser_press,bg="forest green")
        self.epochString = tk.StringVar(value="Episodes: 0")
        self.scoreString = tk.StringVar(value="Score/Best: 0/0")
        self.epochLabel = tk.Label(self.frame,textvariable=self.epochString, relief=tk.RIDGE, padx=5, pady=2)
        self.scoreLabel = tk.Label(self.frame,textvariable=self.scoreString, relief=tk.RIDGE, padx=5, pady=2)
        self.mlLabel = tk.Label(self.frame,text="ML Controls:", relief=tk.RIDGE, padx=5, pady=2)
        self.humanLabel = tk.Label(self.frame,text="Entertainment for humans:", relief=tk.RIDGE, padx=5, pady=2)
        # Customization
        self.labelCustomization = tk.Label(self.frame, text="Customization", relief=tk.RIDGE, padx=5, pady=2)
        # Memory capacity, Batch size, Alpha learning rate, Gamma discount, Epsilon greedy
        self.stringMemorycap = tk.StringVar(value="Mem. capacity: "+str(self.memory_capacity))
        self.labelMemorycap = tk.Label(self.frame,textvariable=self.stringMemorycap, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonMemorycap = tk.Button(self.frame, text="Set", command=self.buttonMemorycap_press, bg="sea green")
        self.stringBatchsize = tk.StringVar(value="Batch size: "+str(self.batch_size))
        self.labelBatchsize = tk.Label(self.frame,textvariable=self.stringBatchsize, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonBatchsize = tk.Button(self.frame, text="Set", command=self.buttonBatchsize_press, bg="sea green")
        self.stringCiters = tk.StringVar(value="C-iters: "+str(self.c_iters))
        self.labelCiters = tk.Label(self.frame,textvariable=self.stringCiters, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonCiters = tk.Button(self.frame, text="Set", command=self.buttonCiters_press, bg="sea green")
        self.stringAlphalr = tk.StringVar(value="α-learning: "+str(self.learning_rate))
        self.labelAlphalr = tk.Label(self.frame,textvariable=self.stringAlphalr, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonAlphalr = tk.Button(self.frame, text="Set", command=self.buttonAlphalr_press, bg="sea green")
        self.stringGammadisc = tk.StringVar(value="γ-discount: "+str(self.discount))
        self.labelGammadisc = tk.Label(self.frame,textvariable=self.stringGammadisc, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonGammadisc = tk.Button(self.frame, text="Set", command=self.buttonGammadisc_press, bg="sea green")
        self.stringEpsilongreedy = tk.StringVar(value="ε-greedy: "+str(self.greedy))
        self.labelEpsilongreedy = tk.Label(self.frame,textvariable=self.stringEpsilongreedy, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonEpsilongreedy = tk.Button(self.frame, text="Set", command=self.buttonEpsilongreedy_press, bg="sea green")
        self.stringEtadecay = tk.StringVar(value="η-decay: "+str(self.decay))
        self.labelEtadecay = tk.Label(self.frame,textvariable=self.stringEtadecay, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonEtadecay = tk.Button(self.frame, text="Set", command=self.buttonEtadecay_press, bg="sea green")
        # Start
        self.root.after(0,self.update_loop)
        self.root.mainloop()
    
    # Resizing event
    def resizing_event(self,event):
        if event.widget == self.root:
            self.redraw = True
            self.canvas_width = max(event.width - 250,1)
            self.canvas_height = max(event.height - 40,1)
            self.frame.configure(width=event.width,height=event.height)
            self.canvas.configure(width=self.canvas_width,height=self.canvas_height)
            self.canvas.place(x=20,y=20)
            if (WIDTH/HEIGHT)*self.canvas_height > self.canvas_width:
                self.board_width,self.board_height = self.canvas_width,int((HEIGHT/WIDTH)*self.canvas_width)
            else:
                self.board_height,self.board_width = self.canvas_height,int((WIDTH/HEIGHT)*self.canvas_height)
            self.board_offset_x,self.board_offset_y = (self.canvas_width - self.board_width)//2,(self.canvas_height - self.board_height)//2
            self.mlLabel.place(x=event.width - 210, y=20, width=200)
            self.epochLabel.place(x=event.width - 210, y=50)
            self.scoreLabel.place(x=event.width - 210, y=80)
            self.buttonReset.place(x=event.width - 195, y=110, width=50)
            self.buttonStep.place(x=event.width - 130, y=110, width=50)
            self.buttonSkip.place(x=event.width - 65, y=110, width=50)
            self.buttonPlayAgent.place(x=event.width - 155, y=145, width=100)
            self.humanLabel.place(x=event.width - 210, y=200, width=200)
            self.buttonPlayUser.place(x=event.width - 155, y=235, width=100)
            # Customization
            self.labelCustomization.place(x=event.width - 210, y=290, width=190)
            self.labelMemorycap.place(x=event.width - 210, y=320)
            self.buttonMemorycap.place(x=event.width - 50, y=320)
            self.labelBatchsize.place(x=event.width - 210, y=350)
            self.buttonBatchsize.place(x=event.width - 50, y=350)
            self.labelCiters.place(x=event.width - 210, y=380)
            self.buttonCiters.place(x=event.width - 50, y=380)
            self.labelAlphalr.place(x=event.width - 210, y=410)
            self.buttonAlphalr.place(x=event.width - 50, y=410)
            self.labelGammadisc.place(x=event.width - 210, y=440)
            self.buttonGammadisc.place(x=event.width - 50, y=440)
            self.labelEpsilongreedy.place(x=event.width - 210, y=470)
            self.buttonEpsilongreedy.place(x=event.width - 50, y=470)
            self.labelEtadecay.place(x=event.width - 210, y=500)
            self.buttonEtadecay.place(x=event.width - 50, y=500)
    
    # Key press event
    def key_press_event(self,event):
        self.direction = (event.keycode - 38)%4
    
    # Update loop
    def update_loop(self):
        if time.time() - self.last_action >= 1/APS:
            if self.playing_user and self.direction>=0:
                self.redraw = True
                self.last_direction = self.direction
                self.last_action = time.time()
                self.snake.perform_action(self.direction)
                if self.snake.is_terminal_state(): self.buttonPlayuser_press()
            elif self.playing_agent:
                self.redraw = True
                self.last_action = time.time()
                if not self.snake.is_terminal_state(): self.agent.step(self.snake, learn=False)
        if self.redraw:
            self.redraw_canvas()
        self.root.after(int(1000/FPS),self.update_loop)
    
    # Play User button
    def buttonPlayuser_press(self):
        if not self.playing_agent:
            self.playing_user = not self.playing_user
            self.buttonPlayUser.configure(text=("Stop" if self.playing_user else "Play"), bg=("indian red" if self.playing_user else "forest green"))
            if self.playing_user:
                self.direction = -1
                self.last_direction = -10
                self.last_action = 0
                self.snake = Snake(seed=int(time.time()*1000))
                self.redraw = True
    
    # Resets the learning model
    def buttonReset_press(self):
        if self.playing_agent or self.playing_user: return
        self.snake = Snake()
        self.agent = self.agentCls(self.memory_capacity, self.batch_size, self.c_iters, self.learning_rate, self.discount, self.greedy, self.decay)
        self.epoch = 0
        self.score,self.highscore = 0,0
        self.epochString.set("Episodes: %i" % (self.epoch,))
        self.scoreString.set("Score/Best: %i/%i" % (self.score,self.highscore))
        self.redraw = True
    
    # Executes an epoch
    def buttonStep_press(self):
        if self.playing_agent or self.playing_user: return
        self.epoch += 1
        self.snake = Snake()
        self.agent.simulation(self.snake)
        self.score = self.snake.get_score()
        self.highscore = max(self.highscore, self.score)
        self.epochString.set("Episodes: %i" % (self.epoch,))
        self.scoreString.set("Score/Best: %i/%i" % (self.score,self.highscore))
        self.redraw = True
    
    # Executes epochs until epoch X
    def buttonSkip_press(self):
        if self.playing_agent or self.playing_user: return
        x = simpledialog.askinteger("Run simulations", "How many simulations:", parent=self.root, minvalue=1, initialvalue=10)
        if x:
            for i in range(x):
                self.buttonStep_press()
                
    # Play Agent button
    def buttonPlayagent_press(self):
        if not self.playing_user:
            self.snake = Snake()
            self.playing_agent = not self.playing_agent                
            self.buttonPlayAgent.configure(text=("Stop" if self.playing_agent else "Simulation"), bg=("indian red" if self.playing_agent else "forest green"))
            self.redraw = True
    
    # Memory-cap button
    def buttonMemorycap_press(self):
        if self.playing_agent or self.playing_user: return
        x = simpledialog.askinteger("Memory capacity", "Input the memory capacity:", parent=self.root,minvalue=1)
        if x:
            self.memory_capacity = x
            self.stringMemorycap.set("Mem. capacity: "+str(self.memory_capacity))
            self.buttonReset_press()
    
    # Batch-size button
    def buttonBatchsize_press(self):
        if self.playing_agent or self.playing_user: return
        x = simpledialog.askinteger("Memory capacity", "Input the batch size:", parent=self.root,minvalue=1)
        if x:
            self.batch_size = x
            self.stringBatchsize.set("Batch size: "+str(self.batch_size))
            self.buttonReset_press()
    
    # C-iters button
    def buttonCiters_press(self):
        if self.playing_agent or self.playing_user: return
        x = simpledialog.askinteger("C-iterations", "Input the c-iterations:", parent=self.root,minvalue=1)
        if x:
            self.c_iters = x
            self.stringCiters.set("C-iters: "+str(self.c_iters))
            self.buttonReset_press()
    
    # Alpha-lr button
    def buttonAlphalr_press(self):
        if self.playing_agent or self.playing_user: return
        x = simpledialog.askfloat("α Learning rate", "Input the learning rate:", parent=self.root,minvalue=0,maxvalue=1)
        if x:
            self.learning_rate = x
            self.stringAlphalr.set("α-learning: "+str(self.learning_rate))
            self.buttonReset_press()
    
    # Gamma-disc button
    def buttonGammadisc_press(self):
        if self.playing_agent or self.playing_user: return
        x = simpledialog.askfloat("γ Discount factor", "Input the discount factor:", parent=self.root,minvalue=0,maxvalue=1)
        if x:
            self.discount = x
            self.stringGammadisc.set("γ-discount: "+str(self.discount))
            self.buttonReset_press()
    
    # Epsilon-greedy button
    def buttonEpsilongreedy_press(self):
        if self.playing_agent or self.playing_user: return
        x = simpledialog.askfloat("ε Greedy", "Input the initial ε greedy value:", parent=self.root,minvalue=0,maxvalue=1)
        if x:
            self.greedy = x
            self.stringEpsilongreedy.set("ε-greedy: "+str(self.greedy))
            self.buttonReset_press()
    
    # Eta-decay button
    def buttonEtadecay_press(self):
        if self.playing_agent or self.playing_user: return
        x = simpledialog.askfloat("η Decay factor", "Input the η decay factor for ε:", parent=self.root,minvalue=0,maxvalue=1)
        if x:
            self.decay = x
            self.stringEtadecay.set("η-decay: "+str(self.decay))
            self.buttonReset_press()
    
    def redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0,0,self.canvas_width,self.canvas_height,fill="#606060",width=0)
        pixels = np.array( [[colors[y] for y in x] for x in self.snake.board] )
        self.image = Image.fromarray(pixels.astype('uint8'), 'RGB')
        self.photo = ImageTk.PhotoImage(image=self.image.resize((self.board_width,self.board_height),resample=Image.NEAREST))
        self.canvas.create_image(self.board_offset_x,self.board_offset_y,image=self.photo,anchor=tk.NW)
        dy = self.board_height / HEIGHT
        dx = self.board_width / WIDTH
        for i in range(1,HEIGHT):
            self.canvas.create_line(self.board_offset_x, self.board_offset_y+int(dy*i), self.board_offset_x+self.board_width,self.board_offset_y+int(dy*i))
        for i in range(1,WIDTH):
            self.canvas.create_line(self.board_offset_x + int(dx*i), self.board_offset_y, self.board_offset_x+int(dx*i),self.board_offset_y+self.board_height)
        self.canvas.create_rectangle(self.board_offset_x,self.board_offset_y,self.board_offset_x+self.board_width,self.board_offset_y+self.board_height,outline="#0000FF",width=3)
        self.redraw = False

if __name__ == "__main__":
    x = mainWindow()
