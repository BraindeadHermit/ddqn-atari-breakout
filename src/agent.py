import torch
import random
import copy
import torch.nn.functional as F
import torch.optim as optim
from plot import Plots 
import numpy as np 

# Memoria dell'agente
class ReplayMemory:
    def __init__(self, capacity=10000, device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0

    '''
        Gestisce la memoria dell'agente, nel momento in cui la memoria ram della gpu è piena, 
        allora i dati vengono passati alla cpu, che li memorizza all'interno della ram del pc
    '''
    def push(self, *args):
        args = [item.to(self.device) for item in args]

        if len(self.memory) < self.capacity:
            self.memory.append(args)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(args)

    def can_provide_sample(self, memory, batch_size):
        return len(self.memory) >= batch_size * 10

    '''
        quando servono, i dati vengono riportati sulla gpu
    '''
    def sample(self, batch_size=32):
        assert self.can_provide_sample(self.memory, batch_size)
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        # riportiamo i dati sulla gpu
        return [torch.cat(item).to(self.device) for item in batch]

    '''
        Ci assicuriamo che la lunghezza della memoria sia accessibile a tutti dall'esterno
        @return: la lunghezza della memoria
    '''
    def __len__(self):
        return len(self.memory)
    
class Agent:
    def __init__(self, model, device='cpu', epsilon=1.0, min_epsilon=0.1, gamma=0.99, nb_warmup=10000, nb_actions=None, memory_size=10000,
                 batch_size=32, learning_rate=0.00025):
        self.memory = ReplayMemory(capacity=memory_size, device=device)
        self.device = device
        self.model = model
        self.target_model = copy.deepcopy(model).to(device)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.nb_warmup = nb_warmup
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / self.nb_warmup
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = gamma
        self.nb_actions = nb_actions

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        print('Agent initialized with device: ', device)
        print('Agent initialized with epsilon: ', epsilon)
        print('Agent initialized with epsilon decay: ', self.epsilon_decay)

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1))
        else:
            av = self.model(state).detach()
            # restituisce una lista di probabilità, argmax si occuperà di prendere quella più alta, e quindi quella
            # che con maggiore probabilità ci darà il reward più alto, e restituirà l'indice di essa nell'array
            return torch.argmax(av, dim=1, keepdim=True)
        
    def train(self, env, epochs):
        stats = {'loss': [], 'reward': [], 'rewards': [], 'avg_reward': [], 'epsilon': []}

        plotter = Plots()

        for epoch in range(epochs):
            # ad ogni epoca resettiamo l'ambiente
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                self.memory.push(state, action, next_state, reward, done)


                if self.memory.can_provide_sample(memory=self.memory, batch_size=self.batch_size):
                    state_b, action_b, next_state_b, reward_b, done_b = self.memory.sample(self.batch_size)
                    # q state action
                    qsa_b = self.model(state_b).gather(1, action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=1, keepdim=True)[0]
                    target_b = reward_b + self.gamma * next_qsa_b *  ~done_b
                    loss = F.smooth_l1_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    # back propagation
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                total_reward += reward.item()
            
            stats['reward'].append(total_reward)

            if self.epsilon > self.min_epsilon:
                self.epsilon -= self.epsilon_decay

            # aggiorniamo il target model
            if epoch % 10 == 0:
                self.model.save_model('model.pth')
                print('Epoch: ', epoch, 'Reward: ', total_reward, 'Epsilon: ', self.epsilon)
                stats['rewards'].append(total_reward)
                stats['avg_reward'].append(np.mean(stats['rewards'][-100:]))
                stats['epsilon'].append(self.epsilon)
                
                '''
                    per le prime 100 iterazioni stampiamo il reward di ogni episodio,
                    per le iterazioni successive stampiamo la media dei reward degli ultimi 100 episodi
                '''
                if (len(stats['rewards'])) > 100:
                    print(f'Epoch: {epoch}, Avg Reward: {np.mean(stats["rewards"][-100:])}, Epsilon: {self.epsilon}')
                else:
                    print(f'Epoch: {epoch}, Avg Reward: {np.mean(stats["rewards"][-1:])}, Epsilon: {self.epsilon}')
            
            # aggiorniamo il target model
            if epoch % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                #plotter.plot(stats)

            '''
                salviamo il modello ogni 1000 epoche,
                il motivo di ciò è che nel momento in cui il modello dovesse raggiungere delle performance buone 
                all'epoca x e poi cominciare a peggiorare, abbiamo il modello a quella determinata epoca salvato
            '''
            if epoch % 1000 == 0:
                self.model.save_model('model/model_{epoch}.pth')
                print('Model saved')
        
        return stats
    
    def test(self, env, epochs):
        stats = {'reward': [], 'rewards': [], 'avg_reward': []}
        for epoch in range(epochs):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state

            stats['reward'].append(total_reward)
            stats['rewards'].append(total_reward)
            stats['avg_reward'].append(np.mean(stats['rewards']))
            print(f'Epoch: {epoch}, Avg Reward: {np.mean(stats["rewards"])}')
        
        return stats