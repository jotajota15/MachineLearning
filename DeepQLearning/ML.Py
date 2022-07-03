import torch as pt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import collections
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
            pt.nn.Softmax()
            )
            self.dropout = pt.nn.Dropout(0.10)

        def forward(self, x):
            x = self.l1(x)
            x = self.dropout(x)
            x = self.l2(x)
            x = self.dropout(x)
            x = self.l3(x)
            x = self.dropout(x)
            self.epoch=+1
            return x.argmax().item()
    
    def __init__(self,learning_rate,memory_capacity,batch_size,c_iters,discount_factor):
        '''
        Main Variables
        '''
        # They have begin at the same time
        self.brain_target = self.Snake_brain()
        self.brain_policy = self.Snake_brain()
        self.optimizer = pt.optim.Adam(self.brain_policy.parameters(), lr=learning_rate)	# Optimizador
        self.loss_fn = pt.nn.MSELoss()
        self.new_target_nn()
        '''
        Other Variables 
        '''
        self.epoch = 0
        self.batch_size = batch_size  
        self.c_iters = c_iters
        self.iterations = 0
        self.discount_factor = discount_factor
        self.memory = dict()
        self.memory["action"] = collections.deque([],memory_capacity)
        self.memory["state"] = collections.deque([],memory_capacity)
        self.memory["reward"] = collections.deque([],memory_capacity)
        #torch.tensor(self.STATS[M])

    # Method that get the batch to train
    def get_batch(self):
        # set memories
        # actions = pt.tensor(self.memory["action"])
        # states = pt.tensor(self.memory["states"])
        # reward = pt.tensor(self.memory["reward"])
        traindata = MyDataset(pt.tensor(self.memory["action"]), pt.tensor(self.memory["states"]),pt.tensor(self.memory["reward"]))
        return DataLoader(traindata, batch_size=self.batch_size, shuffle=True)
    # Method that change the Target NN as Policy NN 
    def new_target_nn(self):
        self.brain_target.load_state_dict(self.brain_policy.state_dict())

    # Method that make the training in Policiy NN 
    def train(self,batch):
        iterator = iter(batch)
        actions,states,rewards = next(iterator) # https://github.com/pytorch/pytorch/issues/1917 TODO: understand actions
        self.optimizer.zero_grad()
        # TODO: This wrong, you have to change only one vector , remember that rewards just affect one remember
        rewards_pred = self.brain_policy(  states[0] )
        with pt.no_grad():			# Apaga el cálculo de gradientes
            rewards_target = rewards +  self.brain_target(states[1]) * self.discount_factor
            rewards_target.detach()
        
        loss = self.loss_fn(rewards_pred, rewards_target)
        loss.backward()
        self.optimizer.step()

    def learn(self):
        # ask for c_iters
        if(self.iterations==self.c_iters):
            #change NN 
            self.new_target_nn(self)
        # train  
        self.train(self.get_batch(self))


    # Do a single best movement TODO: to know if this is nn to use and if has to detach or not
    def best_movement(self,current_state):
        return self.brain_policy(pt.Tensor(current_state))
    # To set a new memory in the agent brain
    def new_memory(self,current_state,reward,new_state,action):
        self.memory["action"].appendleft(action)
        self.memory["state"].appendleft((current_state,new_state))
        self.memory["reward"].appendleft(reward)

       



    def save_model(self,optimizer,best_vloss,loss_train,loss_test,reward):
        pt.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_vloss,
            'loss_train':loss_train,
            'loss_test' : loss_test,
            'reward'    : reward
            }, SAVEFILE)



