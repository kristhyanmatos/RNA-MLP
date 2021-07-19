import numpy as np


class PortaLogicaNOT:
    def __init__(self, peso, bias):
        self.peso = peso
        self.bias = bias

    def mapeamento(self, entrada_balanceada):
        for index, entrada in enumerate(entrada_balanceada):
            if entrada >= 0:
                entrada_balanceada[index] = 1
            else:
                entrada_balanceada[index] = 0
        return entrada_balanceada

    def perceptron(self, entrada):
        entrada_balanceada = np.dot(self.peso, entrada) + self.bias
        return self.mapeamento(entrada_balanceada)


aprendizagem = PortaLogicaNOT(
    bias=0.5,
    peso=-1,
)

entrada = np.array([0, 1, 0, 0])
resposta = aprendizagem.perceptron(entrada)

print("Saída NOT: ", resposta)
