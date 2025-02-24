import torch
from breakout import DQNBreackout
from model import DDQNAgent
from agent import Agent

"""
    Atari Breakout con dueling deep q-learning
"""

# nel momento in cui si esegue il codice, si controlla se è disponibile una GPU, altrimenti si utilizza la CPU del pc
print("Cuda is available: ", torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreackout(device=device, repeat=8, render_mode='human')

model = DDQNAgent(nb_actions=4)
model.to(device)

#model.load_model("model/atari-brekout-v1.0.pth")
model.load_model("model/atari-brekout-v2.0.pth")

EPSILON = 0.05
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
BATCH_SIZE = 64
EPOCHS = 10    
MEMORY_SIZE = 1000000                                        
NB_WARMUP  = 5000
NB_ACTIONS = 4


agent = Agent(model=model, 
              device=device, 
              epsilon=EPSILON, 
              nb_warmup=NB_WARMUP, 
              nb_actions=NB_ACTIONS, 
              learning_rate=LEARNING_RATE,
              memory_size=MEMORY_SIZE,
              batch_size=BATCH_SIZE,
              gamma=GAMMA)

agent.test(env=environment, epochs=EPOCHS)