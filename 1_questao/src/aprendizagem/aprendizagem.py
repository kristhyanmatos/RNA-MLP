import numpy as np
from matplotlib import pyplot as plt


class Aprendizagem:
    def __init__(
        self,
        modelo_entrada,
        modelo_saida,
        numero_neuronios_camada_oculta,
        taxa_aprendizagem,
    ):
        self.modelo_entrada = modelo_entrada
        self.modelo_saida = modelo_saida
        self.taxa_aprendizagem = taxa_aprendizagem
        self.numero_neuronios_camada_oculta = numero_neuronios_camada_oculta
        self.numero_entradas = self.modelo_entrada.shape[0]
        self.numero_saidas = self.modelo_saida.shape[0]
        self.taxa_erro = []
        self.parametros = self.inicializaParametros(
            self.numero_entradas,
            self.numero_neuronios_camada_oculta,
            self.numero_saidas,
        )
        self.aprendeu = False
        self.ultimos_dez_valores_taxa_erro = []
        while self.aprendeu is not True:
            taxa_erro, cache, _ = self.forwardPropagation(
                self.modelo_entrada, self.modelo_saida, self.parametros
            )
            taxa_erro = np.round(taxa_erro, 5)
            self.taxa_erro.append(taxa_erro)
            gradientes = self.backwardPropagation(
                self.modelo_entrada, self.modelo_saida, cache
            )
            self.parametros = self.atualizaParametros(
                self.parametros, gradientes, self.taxa_aprendizagem
            )
            if len(self.ultimos_dez_valores_taxa_erro) >= 10:
                self.ultimos_dez_valores_taxa_erro.pop(0)
                self.ultimos_dez_valores_taxa_erro.append(taxa_erro)

                if len(np.unique(self.ultimos_dez_valores_taxa_erro)) == 1:
                    self.aprendeu = True

            else:
                self.ultimos_dez_valores_taxa_erro.append(taxa_erro)

            # print(self.ultimos_dez_valores_taxa_erro)

        plt.figure()
        plt.plot(self.taxa_erro)
        plt.xlabel("ÉPOCAS")
        plt.ylabel("Taxa de Erro")
        plt.show()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def inicializaParametros(
        self,
        numero_entradas,
        numero_neuronios_camadas_ocultas,
        numero_saidas,
    ):

        pesos_entrada = np.random.randn(
            numero_neuronios_camadas_ocultas, numero_entradas
        )
        pesos_saida = np.random.randn(numero_saidas, numero_neuronios_camadas_ocultas)
        polatrizacao_1 = np.zeros((numero_neuronios_camadas_ocultas, 1))
        polatrizacao_2 = np.zeros((numero_saidas, 1))

        parametros = {
            "pesos_entrada": pesos_entrada,
            "pesos_saida": pesos_saida,
            "bias_1": polatrizacao_1,
            "bias_2": polatrizacao_2,
        }
        return parametros

    def forwardPropagation(self, entrada, saida, parametros):
        numero_entradas = entrada.shape[1]
        pesos_entrada = parametros["pesos_entrada"]
        pesos_saida = parametros["pesos_saida"]
        bias_1 = parametros["bias_1"]
        bias_2 = parametros["bias_2"]

        entrada_balanceada = np.dot(pesos_entrada, entrada) + bias_1
        entrada_utilizada = self.sigmoid(entrada_balanceada)
        saida_balanceada = np.dot(pesos_saida, entrada_utilizada) + bias_2
        saida_obtida = self.sigmoid(saida_balanceada)

        cache = (
            entrada_balanceada,
            entrada_utilizada,
            pesos_entrada,
            bias_1,
            saida_balanceada,
            saida_obtida,
            pesos_saida,
            bias_2,
        )
        logprobs = np.multiply(np.log(saida_obtida), saida) + np.multiply(
            np.log(1 - saida_obtida), (1 - saida)
        )
        taxa_erro = -np.sum(logprobs) / numero_entradas
        return taxa_erro, cache, saida_obtida

    def backwardPropagation(self, entrada, saida, cache):
        numero_entradas = entrada.shape[1]
        (
            entrada_balanceada,
            entrada_utilizada,
            pesos_entrada,
            bias_1,
            saida_balanceada,
            saida_obtida,
            pesos_saida,
            bias_2,
        ) = cache

        erro_saida = saida_obtida - saida
        pesos_saida = np.dot(erro_saida, entrada_utilizada.T) / numero_entradas
        bias_2 = np.sum(erro_saida, axis=1, keepdims=True)

        dA1 = np.dot(pesos_saida.T, erro_saida)
        erro_entrada = np.multiply(dA1, entrada_utilizada * (1 - entrada_utilizada))
        pesos_entrada = np.dot(erro_entrada, entrada.T) / numero_entradas
        bias_1 = np.sum(erro_entrada, axis=1, keepdims=True) / numero_entradas

        gradientes = {
            "pesos_entrada": pesos_entrada,
            "pesos_saida": pesos_saida,
            "bias_1": bias_1,
            "bias_2": bias_2,
            "erro_entrada": erro_entrada,
            "erro_saida": erro_saida,
        }
        return gradientes

    def atualizaParametros(self, parametros, gradientes, taxa_aprendizado):
        parametros["pesos_entrada"] = (
            parametros["pesos_entrada"] - taxa_aprendizado * gradientes["pesos_entrada"]
        )
        parametros["pesos_saida"] = (
            parametros["pesos_saida"] - taxa_aprendizado * gradientes["pesos_saida"]
        )
        parametros["bias_1"] = (
            parametros["bias_1"] - taxa_aprendizado * gradientes["bias_1"]
        )
        parametros["bias_2"] = (
            parametros["bias_2"] - taxa_aprendizado * gradientes["bias_2"]
        )
        return parametros
