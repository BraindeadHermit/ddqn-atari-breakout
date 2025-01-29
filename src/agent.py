import torch
import random
import copy
import torch.nn.functional as F
import torch.optim as optim
from plot import Plots 
import numpy as np 
#import curses
import time
import random

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
        return len(memory) >= batch_size * 10

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
                 batch_size=32, learning_rate=0.00025, tau=0.001):
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
        self.tau = tau

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        print('-----------------------------------------------------------------------------------------------')
        print('Agent initialized with device: ', device)
        print('Agent initialized with epsilon: ', epsilon)
        print('Agent initialized with epsilon decay: ', self.epsilon_decay)
        print('-----------------------------------------------------------------------------------------------')

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1))
        else:
            av = self.model(state).detach()
            # restituisce una lista di probabilità, argmax si occuperà di prendere quella più alta, e quindi quella
            # che con maggiore probabilità ci darà il reward più alto, e restituirà l'indice di essa nell'array
            return torch.argmax(av, dim=1, keepdim=True)
        
    def train(self, env, epochs):
        stats = {'loss': [], 'rewards': [], 'avg_reward': [], 'epsilon': []}

        plotter = Plots(epochs=epochs)

        for epoch in range(0, epochs):
            # ad ogni epoca resettiamo l'ambiente
            state = env.reset()
            done = False
            total_reward = 0
            loss_val = 0.0

            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                self.memory.push(state, action, next_state, reward, done)


                if self.memory.can_provide_sample(memory=self.memory, batch_size=self.batch_size):
                    state_b, action_b, next_state_b, reward_b, done_b = self.memory.sample(self.batch_size)
                    # q state action
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=1, keepdim=True)[0]
                    target_b = reward_b + self.gamma * next_qsa_b *  ~done_b
                    qsa_b = self.model(state_b).gather(1, action_b)
                    loss = F.mse_loss(qsa_b, target_b)
                    self.optimizer.zero_grad()
                    # back propagation
                    loss.backward()
                    self.optimizer.step()
                    self.soft_update()

                    loss_val = loss.cpu().data.numpy()
                    stats['loss'].append(loss_val)
                    print("Loss: ", loss_val)


                state = next_state
                total_reward += reward.item()
            
            print('Epoch Number', epoch)

            stats['rewards'].append(total_reward)
            stats['avg_reward'].append(np.mean(stats['rewards']))
            stats['epsilon'].append(self.epsilon)

            if self.epsilon > self.min_epsilon:
                self.epsilon -= self.epsilon_decay

            # mostriamo le statistiche
            print('Epoch: ', epoch, 'Avg Reward: ', total_reward, 'Epsilon: ', self.epsilon)

            '''
                salviamo il modello ogni 1000 epoche,
                il motivo di ciò è che nel momento in cui il modello dovesse raggiungere delle performance buone 
                all'epoca x e poi cominciare a peggiorare, abbiamo il modello a quella determinata epoca salvato
            '''
            if epoch % 1000 == 0:
                self.model.save_model(f'model/model_{epoch}.pth')
                print('Model saved')
        
        self.model.save_model("model/atari-brekout-v2.0.pth")
        plotter.plot(stats)

        return stats
    
    def soft_update(self):
        # Soft update
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def test(self, env, epochs):
        for _ in range(0, epochs):
            state = env.reset()
            done = False
    
            while not done:
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, _, _ = env.step(action)

    """
    def render_table(stdscr, data, start_row=0, start_col=0):
        
            Rendeizza 'data' come una piccola tabella a partire dalle coordinate
            (start_row, start_col) sul terminale curses.
    
            data: lista di liste, dove data[0] è l'intestazione e
            data[i][j] è il contenuto della cella i,j
        
        for row_idx, row_data in enumerate(data):
            for col_idx, cell in enumerate(row_data):
                # Spostiamo ogni colonna di 12 caratteri per non sovrapporre i valori
                stdscr.addstr(start_row + row_idx, start_col + col_idx*12, str(cell))

    def update_display(self, stdscr, data, current_loss, epoch):
        
        #    - Pulisce lo schermo
         #   - Mostra la tabella 'data' (Epoch, Accuracy, Val_Loss, ...)
          #  - Mostra la loss corrente in una posizione dedicata
           # - Mostra l'epoch corrente o altre info
            #- Aggiorna l'output su terminale
        
        stdscr.clear()
    
        # Render della tabella in alto (partendo dalla riga 0, colonna 0)
        self.render_table(stdscr, data, start_row=0, start_col=0)
    
        # Mostriamo la loss e l'epoca sotto la tabella
        table_height = len(data)
        stdscr.addstr(table_height + 2, 0, f"Epoch: {epoch}")
        stdscr.addstr(table_height + 3, 0, f"Current Loss: {current_loss:.6f}")
    
        # Applichiamo le modifiche
        stdscr.refresh()
    
    if __name__ == "__main__":
        curses.wrapper(train)
       """         