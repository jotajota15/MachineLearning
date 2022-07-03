import torch as pt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import namedtuple, deque
SAVEFILE = "model"

class MyDataset(Dataset):
    def __init__(self, x, y,z=None):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], None if self.z == None else self.z[index]

class Snake_learning():

    

    class ReplayMemory(object):
        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

        def __init__(self, capacity):
            self.memory = deque([],maxlen=capacity)


        def push(self, *args):
            """Save a transition"""
            self.memory.append(self.Transition(*args))

        def sample(self, batch_size,random):
            transitions = random.sample(self.memory, batch_size)
            return self.Transition(*zip(*transitions))

        def __len__(self):
            return len(self.memory)

    class Snake_brain(pt.nn.Module):
        def __init__(self):
            '''
            Se crea la red neuronal
            '''
            super().__init__()
            self.l1 = pt.nn.Sequential(
            pt.nn.Linear(9, 50),
            pt.nn.LeakyReLU()
            )
            self.l2 = pt.nn.Sequential(
            pt.nn.Linear(50, 100),
            pt.nn.ReLU()
            )
            self.l3 = pt.nn.Sequential(
            pt.nn.Linear(100, 3),
            pt.nn.ReLU()
            )
            self.dropout = pt.nn.Dropout(0.10)

        def forward(self, x):
            if x.shape[0]!= 9:
                a = 2
            x = self.l1(x)
            x = self.dropout(x)
            x = self.l2(x)
            x = self.dropout(x)
            x = self.l3(x)
            x = self.dropout(x)
            return x
    
    def __init__(self,learning_rate,memory_capacity,batch_size,c_iters,discount_factor):
        '''
        Main Variables
        '''
        # They have begin at the same time
        self.brain_target = self.Snake_brain()
        self.brain_policy = self.Snake_brain()
        self.loss_fn = pt.nn.SmoothL1Loss() # You can change it
        self.optimizer = pt.optim.RMSprop(self.brain_policy.parameters(), lr=learning_rate) 
        # self.loss_fn = pt.nn.MSELoss()
        # self.optimizer = pt.optim.Adam(self.brain_policy.parameters(), lr=learning_rate)	# Optimizador
        # self.new_target_nn()
        '''
        Other Variables 
        '''
        self.epoch = 0
        self.batch_size = batch_size  
        self.c_iters = c_iters
        self.iterations = 0
        self.discount_factor = discount_factor
        self.memory = self.ReplayMemory(memory_capacity)
        self.memory_capacity = memory_capacity
        #torch.tensor(self.STATS[M])

    # Method that get the batch to train
    def get_batch(self,prng):
        if len(self.memory) < self.memory_capacity:
            return None
        return self.memory.sample(self.batch_size,prng)
    
    # Method that change the Target NN as Policy NN 
    def new_target_nn(self):
        self.brain_target.load_state_dict(self.brain_policy.state_dict())

    # Method to optimize the model
    def optimize(self,loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Juan_Problem: Look for this change , this is not natural, this can be a problem 
        for param in self.brain_policy.parameters():
            param.grad.data.clamp_(-1, 1) # https://deepai.org/machine-learning-glossary-and-terms/gradient-clipping#:~:text=Gradient%20clipping%20is%20a%20technique,input%20into%20a%20specific%20output.
        self.optimizer.step()

    # Method that make the training in Policiy NN 
    def train(self,batch,current_apples):
        # Get other batch forms
        self.epoch+=1
        non_final_mask = pt.tensor(tuple(map(lambda s: s is not None, # Lógica de estados finales 
                                          batch.next_state)),dtype=pt.bool)#,device=device) # Pone si es o no normal, ayuda en el reward
        non_final_next_states = pt.cat([s for s in batch.next_state
                                                if s is not None]) # Mete estados que no son finales
        state_batch = pt.cat(batch.state).reshape((self.batch_size,9)) # Le quita la tupla
        action_batch = pt.cat(batch.action).reshape(1,self.batch_size)
        reward_batch = pt.cat(batch.reward)
        state_action_values = self.brain_policy(state_batch).gather(1, action_batch) # Juan Gather makes the magic GatherBackward0
        next_state_values = pt.zeros(self.batch_size) #device=device) # Juan If is not state not update is made
        next_state_values[non_final_mask] = self.brain_target(non_final_next_states).max(1)[0].detach() # Chose better and detach

        # Compute the expected Q values.
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch # Ojo que aunque este en 0, el reward siempre juega
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(0)) # Esto esta raro que este 1, 50 TODO
        self.optimize(loss)
        self.save_model(loss.item(),current_apples)
        
    def learn(self,prng,current_apples):
        # ask for c_iters        
        if(self.iterations==self.c_iters):
            #change NN 
            self.new_target_nn(self)
        # train  
        self.c_iters +=1
        batch = self.get_batch(prng)
        if batch is None:
            return 
        self.train(batch,current_apples)


    # Do a single best movement TODO: to know if this is nn to use and if has to detach or not
    def best_movement(self,current_state):
        with pt.no_grad():
            return self.brain_policy(pt.Tensor(current_state)).argmax().item()

    # To set a new memory in the agent brain
    def new_memory(self,state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

       



    def save_model(self,loss,current_apples):
        pt.save({
            'epoch': self.epoch,
            'model_state_dict': self.brain_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'current_apples': current_apples
            }, SAVEFILE)



