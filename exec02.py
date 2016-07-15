import numpy as np
import Neural_Network as NN
import matplotlib.pyplot as plt
import PokeTrainer as pkt
import validaGradientes as vlg
import csv
import os

# usando dados das bolsas da Argentina e do Chile
###########Parametros############
tamOutput = 1
tamInput = 2
tamCamadaEsc = 3
tamCamadaSaida = 1
###########Leitura das tabelas##########

f = csv.reader(open(os.getcwd()+'\index-data\MERV.csv'), delimiter=',')
merv = []
for linha in f:
	merv.append(float(linha[4]))
merv = merv/np.amax(merv, axis= 0)

############formatação dos dados#############
merv = np.reshape(merv, (len(merv), tamOutput))
#############################################

f = csv.reader(open(os.getcwd()+'\index-data\IPSA.csv'), delimiter=',')
ipsa = []
for linha in f:
	ipsa.append(float(linha[4]))
ipsa = ipsa/np.amax(ipsa, axis= 0)

############formatação dos dados#############
ipsa = np.reshape(ipsa, (len(ipsa), tamOutput))
#############################################

f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
Y = []
for linha in f:
	Y.append(float(linha[4]))
Y = Y/np.amax(Y, axis= 0)

############formatação dos dados#############
Y = np.reshape(Y, (len(Y), tamOutput))
#############################################

TesteNN = NN.Neural_Network(tamInput, tamCamadaSaida, tamCamadaEsc)
Ypredito = TesteNN.propaga(merv, ipsa)
print("Y predito inicial: ", Ypredito)
T = pkt.Treinador(TesteNN)
#0.5 parece ser um bom passo
TesteNN = T.treinar(merv, ipsa, Y, 0.9, 1000, 0.001)
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

