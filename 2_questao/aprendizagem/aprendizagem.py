import pandas
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split


class AprendizagemVinho:
    def __init__(self, numero_epocas) -> None:
        self.dados = pandas.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
        )
        self.entradas = self.dados[self.dados.columns[1:]].to_numpy()
        self.saidas = self.dados[self.dados.columns[0]]

        self.valores_saida_formatados = np.empty((177, 1), dtype=int)

        for index in range(177):
            self.valores_saida_formatados[index] = self.saidas[index] - 1

        (
            self.entradas_treino,
            self.entradas_teste,
            self.saidas_treino,
            self.saidas_teste,
        ) = train_test_split(
            self.entradas,
            self.valores_saida_formatados,
            test_size=0.3,
        )

        self.modelo = keras.Sequential(
            [
                keras.layers.Dropout(0.2),
                keras.layers.Dense(130, activation=tensorflow.nn.relu),
                keras.layers.Dense(70, activation=tensorflow.nn.relu),
                keras.layers.Dense(40, activation=tensorflow.nn.relu),
                keras.layers.Dense(13, activation=tensorflow.nn.relu),
                keras.layers.Dense(3, activation=tensorflow.nn.softmax),
            ]
        )
        self.modelo.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics="accuracy",
        )

        self.hist = self.modelo.fit(
            self.entradas_treino,
            self.saidas_treino,
            epochs=numero_epocas,
            validation_split=0.3,
        )

        plt.plot(self.hist.history["accuracy"])
        plt.plot(self.hist.history["val_accuracy"])
        plt.title("Acurácia por épocas")
        plt.xlabel("Épocas")
        plt.ylabel("Acurácia")
        plt.legend(["Treino", "Valores de Teste"])
        plt.show()

        plt.plot(self.hist.history["loss"])
        plt.plot(self.hist.history["val_loss"])
        plt.title("Taxa de erro por época")
        plt.xlabel("Épocas")
        plt.ylabel("Taxa de erro")
        plt.legend(["Treino", "Valores de Teste"])
        plt.show()
