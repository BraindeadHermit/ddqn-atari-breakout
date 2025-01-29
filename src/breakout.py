import collections
import cv2
import gym
import numpy as np
from PIL import Image
import torch

class DQNBreackout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', repeat = 4, device='cpu'):
        env = gym.make('Breakout-v4',  render_mode=render_mode)
        super(DQNBreackout, self).__init__(env) # eredita da gym.Wrapper, ovvero dall'ambiente iniziale di gym
        self.device = device
        self.repeat = repeat
        self.lives = env.ale.lives()
        self.frame_buffer =  [] # collections.deque(maxlen=repeat)
        self.image_shape = (84, 84)


    """
        Riduzione della complessità dei dati dai quale la rete deve imparare
        riduciamo la dimensione dell'immagine da 210x160 a 84x84
        la renderizziamo in scala di grigi

    """
    def process_obs(self, obs):

        # trasforma l'immagine in scala di grigi
        img = Image.fromarray(obs)
        # ridimensiona l'immagine e la converte in scala di grigi
        img = img.resize(self.image_shape).convert('L')
        img = np.array(img)
        img = torch.from_numpy(img)
        # aggiungi due dimensioni al tensore
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.float() / 255.0

        img = img.to(self.device)

        return img

    # funzione che esegue l'azione dell'agente all'intero dell'ambiente, essa restituisce informazioni sull'ambiente, le osservazioni effettuate
    def step(self, action):
        total_reward = 0.0
        done = False

        """
            andare a considerare ogni frame del nostro ambiente sarebbe superfluo, in quanto anche un umano
            nel momento in cui deve reagire ad un evento all'interno del gioco, esso non reagisce "ad ogni frame",
            ma comunque per il tempo di reazzione umano passano diversi frame,
            cerchiamo di emulare la stessa cosa anche per l'agente, adiamo a dire che per ogni @repeat frame,
            L'agente va a cosiderare la stessa azione.
        """
        for i in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)

            #ad ogni iterazione andiamo ad effettuare la somma dei reward per ottenere il reward totale dell'azione
            total_reward += reward

            # decrement lifes
            current_lives = info['lives'] 

            """
                nel momento in cui l'agente perde una vita durante il gioco,
                fornisci un reward negativo
            """
            if current_lives < self.lives:
                # reward negativo
                total_reward = total_reward - 1
                # aggorna il numero di vite
                self.lives = current_lives

            self.frame_buffer.append(obs)

            if done:
                break

        
        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_obs(max_frame)
        max_frame = max_frame.to(self.device)

        # per essere sicuri che gli output relativi siano restituiti in base al dispositivo che si sta utilizzando (cpu o gpu)
        total_reward = torch.tensor(total_reward).view(1, -1).float().to(self.device)

        done = torch.tensor(done).view(1, -1).to(self.device)
        
        return max_frame, total_reward, done, truncated, info
    
    # reset dell'ambiente, setup iniziale
    def reset(self):
        # clear the buffer
        self.frame_buffer = []

        # reset the observations
        obs, _ = self.env.reset()

        # reset lives
        self.lives = self.env.ale.lives()
        obs = self.process_obs(obs)

        return obs