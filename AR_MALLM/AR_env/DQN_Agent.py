import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
from collections import deque

class ReplayBuffer:
    def __init__(self, config):
        self.buffer_size = config['dqn_config']['buffer_size']
        self.batch_size = config['dqn_config']['batch_size']
        self.buffer = deque(maxlen=self.buffer_size)  # 使用双端队列数据结构实现固定大小的缓冲区
        
    def put(self, transition):
        self.buffer.append(transition)  # 将一条经验(transition)(state, action, reward, next_state, done)存入缓冲区
        
    def sample(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        
        # Convert lists to numpy arrays first
        s_lst = np.array([transition[0] for transition in mini_batch])
        a_lst = np.array([[transition[1]] for transition in mini_batch])  # Keep as 2D array
        r_lst = np.array([[transition[2]] for transition in mini_batch])
        # r_lst = (r_lst - np.mean(r_lst)) / (np.std(r_lst) + 1e-8)
        s_prime_lst = np.array([transition[3] for transition in mini_batch])
        done_lst = np.array([[transition[4]] for transition in mini_batch])
        
        # Convert numpy arrays to torch tensors
        return (torch.FloatTensor(s_lst),
                torch.LongTensor(a_lst),
                torch.FloatTensor(r_lst),
                torch.FloatTensor(s_prime_lst),
                torch.FloatTensor(done_lst))
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, n_state, n_action):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(n_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_action)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.fc3.out_features-1)
        else:
            out = self.forward(obs)
            return out.argmax().item()

def train(q, q_target, memory, optimizer, runner=None):
    gamma = 0.99
    
    for i in range(10):
        s, a, r, s_prime, done = memory.sample()
        
        q_out = q(s)
        q_a = q_out.gather(1, a)
        
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * (1 - done)
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ✅ 新增：记录 loss
        if runner is not None:
            runner.loss_history.append(loss.item())