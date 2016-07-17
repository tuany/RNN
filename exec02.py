import numpy as np
import Neural_Network as NN
import matplotlib.pyplot as plt
import PokeTrainer as pkt
import csv
import os

###########Parametros############
tamOutput = 1
tamInput = 2
tamCamadaEsc = 5
tamCamadaSaida = 1
###########Leitura das tabelas##########

f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
ibovespa = []
for linha in f:
	ibovespa.append(float(linha[4]))
ibovespa = ibovespa/np.amax(ibovespa, axis= 0)

f = csv.reader(open(os.getcwd()+'\index-data\MERV.csv'), delimiter=',')
merv = []
for linha in f:
	merv.append(float(linha[4]))
merv = merv/np.amax(merv, axis= 0)

f = csv.reader(open(os.getcwd()+'\index-data\IPSA.csv'), delimiter=',')
ipsa = []
for linha in f:
	ipsa.append(float(linha[4]))
ipsa = ipsa/np.amax(ipsa, axis= 0)

############# Slices em conjunto treinamento/teste ####################
slice = round(0.66 * len(ibovespa))

conjTreino = np.matrix([merv[:slice], ipsa[:slice]])
conjTeste = np.matrix([merv[slice:], ipsa[slice:]])

############formatação dos dados#############
Ytreino = np.matrix([ibovespa[:slice]])
Yteste = np.matrix([ibovespa[slice:]])

np.delete(conjTreino,-1)
np.delete(Ytreino,0)
np.delete(conjTeste,-1)
np.delete(Yteste,0)

#################################################################
TesteNN = NN.Neural_Network(tamInput, tamCamadaSaida, tamCamadaEsc)
Ytreinopredito = TesteNN.propaga(conjTreino)

print("Ytreino predito inicial: ", Ytreinopredito)
T = pkt.Treinador(TesteNN)
#0.5 parece ser um bom passo
TesteNN = T.treinar(conjTreino, Ytreino, 0.05, 100, 0.0005)
Ytreinopredito = TesteNN.propaga(conjTreino)
print("Ytreino predito final: ", Ytreinopredito)

plt.plot(T.J, 'r-', linewidth=2.0)
plt.grid(1)
plt.show()

#############Testando no conjunto teste#############

YtestePredito = TesteNN.propaga(conjTeste)
print("Yteste real: ", Yteste)
print("Yteste predito final: ", YtestePredito)