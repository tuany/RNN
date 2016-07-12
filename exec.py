import numpy as np
import Neural_Network as NN
import matplotlib.pyplot as plt
import PokeTrainer as pkt
import csv

print(__doc__)
f = csv.reader(open('dados-treinamento\IBOVESPA.csv'), delimiter=',')
valorFechamento = []
for linha in f:
	valorFechamento.append(float(linha[4]))
#print(valorFechamento)

valorFechamento = valorFechamento/np.amax(valorFechamento, axis= 0)
#print(valorFechamento)

f = csv.reader(open('dados-treinamento\IBOVESPA.csv'), delimiter=',')
Y = []
for line in f:
	Y.append(float(line[4]))
Y = Y/np.amax(Y, axis= 0)

TesteNN = NN.Neural_Network(1, 1, 3)

Ypredito = TesteNN.propaga(valorFechamento)


"""
testValues = np.arange(-5,5,0.01)
plt.plot(testValues, TesteNN.sigmoide(testValues), linewidth=2)
plt.plot(testValues, TesteNN.derivadaSigmoide(testValues), linewidth=2)
plt.grid(1)
plt.legend(['sigmoide', 'derivadaSigmoide'])
plt.show()
"""
T = pkt.Treinador(TesteNN)
T.treinar(valorFechamento, Y, 0.9, 100, 0.001)

plt.plot(T.J)
plt.grid(1)
plt.show()
