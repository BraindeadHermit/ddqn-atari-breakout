import matplotlib.pyplot as plt

class Plots():
    def __init__(self, epochs=0):
        self.data = None
        self.eps_data = None
        self.loss = None
        self.epochs = epochs

    '''
        Questa classe si occupa di plottare i grafici
    '''
    def plot(self, stats):
        plt.figure(figsize=(18, 8))
        
        self.data = stats['avg_reward']
        self.eps_data = stats['epsilon']

        plt.xlim(0, self.epochs)

        plt.subplot()
        plt.title('Andamento dell\'addestramento ogni 10 epoche')
        plt.plot(self.data, 'r-', label='Rewards')
        plt.plot(self.eps_data, 'b-', label='Epsilon')
        plt.xlabel('Epoca (x10)')
        plt.ylabel('Reward') 
        plt.legend()

        plt.tight_layout()
        plt.show()