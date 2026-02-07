import numpy as np
from neural_network import NeuralNetwork

def main():
    # 1. Preparação dos Dados (Problema XOR)
    # A terceira coluna é o bias (entrada fixa em 1)
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    y = np.array([[0], [1], [1], [0]])

    # 2. Inicialização do Modelo
    brain = NeuralNetwork(learning_rate=0.5, momentum=0.9)

    # 3. Treinamento
    print("Iniciando treinamento da Rede Neural...")
    brain.train(X, y, epocas=60000)

    # 4. Visualização dos Resultados
    print("\n" + "="*30)
    print("RESULTADOS FINAIS:")
    predicoes = brain.predict(X)
    print(f"Saídas (arredondadas):\n{np.round(predicoes, 2)}")
    print(f"Valores esperados:\n{y}")
    print("="*30)

    # 5. Persistência: Salvando os pesos treinados
    brain.save_weights("pesos_xor_0.npy", "pesos_xor_1.npy")

    print("\nCriando nova instância para testar o carregamento de pesos...")
    nova_rede = NeuralNetwork()
    nova_rede.load_weights("pesos_xor_0.npy", "pesos_xor_1.npy")
    
    # Teste de predição individual manual
    teste_manual = np.array([[1, 0]])
    resultado = nova_rede.predict(teste_manual)
    print(f"Teste manual [1, 0]: {np.round(resultado, 4)}")

    # 6. Exibição da Curva de Aprendizado
    brain.plot_loss()

if __name__ == "__main__":
    main()
