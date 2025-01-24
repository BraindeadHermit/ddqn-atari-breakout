import torch
import random
import copy
import torch.functional as F

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