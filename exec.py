import numpy as np
import Neural_Network as NN
import matplotlib.pyplot as plt
import PokeTrainer as pkt
#import validaGradientes as vlg
import csv
import os

###########Parametros############
tamOutput = 1
tamInput = 1
tamCamadaEsc = 3
tamCamadaSaida = 1
###########Leitura das tabelas##########

f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
dadosFechamento = []
for linha in f:
	dadosFechamento.append(float(linha[4]))
dadosFechamento = dadosFechamento/np.amax(dadosFechamento, axis= 0)

############# Slices em conjunto treinamento/teste ####################
slice = round(0.66 * len(dadosFechamento))

conjTreino = dadosFechamento[:slice]
conjTeste = dadosFechamento[slice:]

############formatação dos dados#############
Ytreino = conjTreino[:]
Yteste = conjTeste[:]

np.delete(conjTreino,-1)
np.delete(Ytreino,0)
np.delete(conjTeste,-1)
np.delete(Yteste,0)

Ytreino = np.reshape(Ytreino, (len(Ytreino),tamOutput))
conjTreino = np.reshape(conjTreino, (len(conjTreino),tamInput))
Yteste = np.reshape(Yteste, (len(Yteste),tamOutput))
conjTeste = np.reshape(conjTeste, (len(conjTeste),tamInput))

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


#testando os gradientes
# grad = TesteNN.computaGradientes(dadosFechamento, Ytreino)
# numgrad = vlg.validaGradientes(TesteNN, dadosFechamento, Ytreino)
# print(grad)
# print("______")
# print(numgrad)

