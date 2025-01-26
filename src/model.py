import torch
import torch.nn as nn
import os

class DDQNAgent(nn.Module):
    '''
        @nb_actions: numero di azioni che prende la rete
    '''
    def __init__(self, nb_actions=4):
        super(DDQNAgent, self).__init__()

        self.relu = nn.ReLU()
        '''
            in_channel: 1 solo canale un quanto trasformiamo l'immagine in scala di grigi,
            i canali dell'immagine passata come output sono 32, quindi il secondo layer prende in inout un'immagine 
            a 32 bit, e come output restituisce un' immagine a 64, che viene presa in input dal prossimo layer, che 
            di conseguenza prende in input un'immagine a 64 bit
            
        '''
        # in questa parte andiamo a definire i layer della nostra rete neurale, che andrà a prendere in input un'immagine
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=32, kernel_size=(8 , 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(4 , 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(3 , 3), stride=(1, 1))

        self.flatten = nn.Flatten()

        # restituisce un detereminato numero di output => in questo caso 20% di probabilità di dropout
        self.dropout = nn.Dropout(p=0.2)

        # dueling network
        # mano mano che l'immagine va avanti
        # 1024 è il numero di neuroni del layer
        '''
            da qui partirà l'analisi delle immagini passate in input
            che restiturià alla fine la probabilità di eseguire una determinata azione
            in particolare questa sezione restituità una lista di probabilità di eseguire una determinata azione
        '''
        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, nb_actions)


        '''
            mentre qui verrà restituita una verrà restituito un valore pesato
        '''
        self.state_value1 = nn.Linear(3136, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)

    '''
        @x: immagine in input al modello
    '''
    def forward(self, x):
        # per sicurezza verifichiamo che l'immagine in input sia un tensore
        x = torch.tensor(x, dtype=torch.float32)
        # Estrazione delle features
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # appiattisce il vettore multidimensionale di output in un vettore monodimensionale
        x = self.flatten(x)

        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)

        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)

        state_value = self.relu(self.state_value3(state_value))

        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)

        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)

        action_value = self.action_value3(action_value)

        # restituisce la probabilità di eseguire una determinata azione
        # va a rappresentare la somma tra il valore dello stato e il valore dell'azione
        return state_value + (action_value - action_value.mean()) 
    
    def save_model(self, path="model/model.pth"):
        # salviamo il dizionario di stati e pesi del modello
        if not os.path.exists("model"):
            os.makedirs("model")
        torch.save(self.state_dict(), path)

    def load_model(self, path="model/model.pth"):
        # carichiamo il dizionario di stati e pesi del modello
        try:
            self.load_state_dict(torch.load(path))
            print("[load] - ","Model loaded successfully")
        except:
            print("[load] - ","Model not loaded")