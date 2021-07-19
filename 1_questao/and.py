import numpy as np
from src.aprendizagem.aprendizagem import Aprendizagem

aprendizagem = Aprendizagem(
    modelo_entrada=np.array([[0, 0, 1, 1], [0, 1, 0, 1]]),
    modelo_saida=np.array([[0, 0, 0, 1]]),
    numero_neuronios_camada_oculta=3,
    taxa_aprendizagem=0.01,
)
entrada = np.array([[1, 1, 0, 0], [0, 1, 0, 1]])
_, _, saida_obtida = aprendizagem.forwardPropagation(
    entrada, aprendizagem.modelo_saida, aprendizagem.parametros
)
resposta = (saida_obtida > 0.5) * 1.0
print("Saída AND:", resposta)
