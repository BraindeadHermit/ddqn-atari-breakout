import torch
from src.breakout import DQNBreackout
from src.model import DDQNAgent

"""
    Atari Breakout con dueling deep q-learning
     
"""

# nel momento in cui si esegue il codice, si controlla se è disponibile una GPU, altrimenti si utilizza la CPU del pc
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreackout(device=device, render_mode='human')

agent = DDQNAgent(nb_actions=4)
agent.to(device)

# resetta l'ambiente dopo ogni iterazione
state = environment.reset()

print(agent.forward(state))

'''
for i in range(100):
    environment.step(environment.action_space.sample())

    state, reward, done, truncated, info = environment.step(environment.action_space.sample())

    # renderizza l'ambiente
    environment.render()
'''