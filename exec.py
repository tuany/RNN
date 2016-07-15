import numpy as np
import Neural_Network as NN
import matplotlib.pyplot as plt
import PokeTrainer as pkt
import validaGradientes as vlg
import csv
import os

###########Parametros############
tamOutput = 1
tamInput = 1
tamCamadaEsc = 3
tamCamadaSaida = 1
###########Leitura das tabelas##########

f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
valorFechamento = []
for linha in f:
	valorFechamento.append(float(linha[4]))
valorFechamento = valorFechamento/np.amax(valorFechamento, axis= 0)

############formatação dos dados#############
Y = valorFechamento[:]
np.delete(valorFechamento,-1)
np.delete(Y,0)
Y = np.reshape(Y, (len(Y),tamOutput))
valorFechamento = np.reshape(valorFechamento, (len(valorFechamento),tamOutput))

#############################################

TesteNN = NN.Neural_Network(tamInput, tamCamadaSaida, tamCamadaEsc)
Ypredito = TesteNN.propaga(valorFechamento)
print("Y predito inicial: ", Ypredito)
T = pkt.Treinador(TesteNN)
#0.5 parece ser um bom passo
TesteNN = T.treinar(valorFechamento, Y, 0.9, 1000, 0.001)
Ypredito = TesteNN.propaga(valorFechamento)
print("Y predito final: ", Ypredito)

plt.plot(T.J, 'r-', linewidth=2.0)
plt.grid(1)
plt.show()

#testando os gradientes
# grad = TesteNN.computaGradientes(valorFechamento, Y)
# numgrad = vlg.validaGradientes(TesteNN, valorFechamento, Y)
# print(grad)
# print("______")
# print(numgrad)

