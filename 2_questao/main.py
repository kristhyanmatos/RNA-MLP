import numpy as np
import matplotlib.pyplot as plt
from aprendizagem.aprendizagem import AprendizagemVinho

aprendizagem = AprendizagemVinho(numero_epocas=400)
taxa_perda, acuracia = aprendizagem.modelo.evaluate(
    aprendizagem.entradas_teste,
    aprendizagem.saidas_teste,
)
saidas_obtidas = aprendizagem.modelo.predict(aprendizagem.entradas_teste)

saidas_formadata = []
for saida in saidas_obtidas:
    saidas_formadata.append(np.argmax(saida))

plt.plot(saidas_formadata)
plt.plot(aprendizagem.saidas_teste)
plt.title("Comparação de Saídas")
plt.xlabel("Posições")
plt.ylabel("Valores de Teste")
plt.legend(["Saídas Obtidas", "Saídas de Teste"])
plt.show()
