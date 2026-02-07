import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self, learning_rate=0.5, momentum=0.9):
        
        # Hiperparâmetros:
        self.lr = learning_rate
        self.mu = momentum
        self.loss_history = [] 
        
        # Inicialização dos pesos (incluindo bias)
        self.weights0 =  2 * np.random.random((3,3)) - 1
        self.weights1 =  2 * np.random.random((4,1)) - 1
        
        # Inicialização das velocidades para o momentum
        self.v_pesos0 = np.zeros_like(self.weights0)
        self.v_pesos1 = np.zeros_like(self.weights1)
        
    # Funções necessárias para calcular deltas e gradientes    
    def _sigmoid(self, soma):
        return 1/(1 + np.exp(-soma))
    
    def _derivada_sigmoid(self, sig):
        return sig * (1-sig)
    
    def _derivada_erro(self, saidas, output):
        return -2*(saidas - output)
        
    def _feed_forward(self, entradas):
        
        # Ativação da camada oculta
        self.hidden_z = np.dot(entradas, self.weights0)
        self.hidden_activation = self._sigmoid(self.hidden_z)
        
        # Adicionando o bias da camada oculta
        self.hidden_bias = np.ones((self.hidden_activation.shape[0], 1))
        self.hidden_activation_and_bias = np.hstack((self.hidden_activation, self.hidden_bias))
        
        # Ativação da camada de saída
        self.output_z = np.dot(self.hidden_activation_and_bias, self.weights1)
        self.output_activation = self._sigmoid(self.output_z)
        
        return self.output_activation
    
    def _backpropagation(self, entradas, saidas):
        
        # Cálculo do delta da camada de saída (derivadas -> regra da cadeia)
        self.d_erro_out = self._derivada_erro(saidas, self.output_activation)
        self.d_out_soma = self._derivada_sigmoid(self.output_activation)
        self.delta_out = self.d_erro_out * self.d_out_soma
        
        # Propagação do erro e delta da camada oculta
        self.erro_repassado = np.dot(self.delta_out, self.weights1.T)
        self.d_hidden_z = self._derivada_sigmoid(self.hidden_activation)
        self.delta_hidden = self.erro_repassado[:, :3] * self.d_hidden_z  
        
        # Gradientes
        self.partial_derivatives_w0 = np.dot(entradas.T, self.delta_hidden)
        self.partial_derivatives_w1 = np.dot(self.hidden_activation_and_bias.T, self.delta_out)
        
        return self.partial_derivatives_w0, self.partial_derivatives_w1
    
    def train(self, entradas, saidas, epocas):
        for i in range(epocas):
            self._feed_forward(entradas)
            grad0, grad1 = self._backpropagation(entradas, saidas)
            
            # Atualização dos pesos
            self.v_pesos0 = (self.mu * self.v_pesos0) + (self.lr * grad0) 
            self.weights0 -= self.v_pesos0
            
            self.v_pesos1 = (self.mu * self.v_pesos1) + (self.lr * grad1) 
            self.weights1 -= self.v_pesos1
            
            # Monitoramento do erro para visualização
            if i % 100 == 0:
                mse = np.mean(np.square(saidas - self.output_activation))
                self.loss_history.append(mse)
            
            if i % 10000 == 0:
                print(f"Erro médio na época {i}: {np.mean(np.abs(saidas - self.output_activation))}")
                
    def save_weights(self, file_w0="weights0.npy", file_w1="weights1.npy"):
        """Salva as matrizes de pesos em formato binário .npy"""
        np.save(file_w0, self.weights0)
        np.save(file_w1, self.weights1)
        print(f"Pesos salvos em {file_w0} e {file_w1}")

    def load_weights(self, file_w0="weights0.npy", file_w1="weights1.npy"):
        """Carrega as matrizes de pesos de arquivos existentes"""
        self.weights0 = np.load(file_w0)
        self.weights1 = np.load(file_w1)
        print("Pesos carregados com sucesso!")

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, color='royalblue', linewidth=2)
        plt.title("Histórico de Erro (MSE) - Convergência da Rede")
        plt.xlabel("Épocas (x100)")
        plt.ylabel("Erro Médio Quadrático")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show() 
                
    def predict(self, entradas):
        if entradas.shape[1] == 2:
            entradas = np.hstack((entradas, np.ones((entradas.shape[0], 1))))
        return self._feed_forward(entradas)
        
        
        