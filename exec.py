import numpy as np
import Neural_Network as NN
import matplotlib.pyplot as plt
import PokeTrainer as pkt
import csv
import os

###########Parametros############
tamOutput = 1
tamInput = 1
tamCamadaEsc = 3
tamCamadaSaida = 1
lambdaVal = 0.00001
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

Ytreino = np.matrix([Ytreino])
conjTreino = np.matrix(conjTreino)
Yteste = np.matrix(Yteste)
conjTeste = np.matrix(conjTeste)

#################################################################
TesteNN = NN.Neural_Network(tamInput, tamCamadaSaida, tamCamadaEsc, lambdaVal)
Ytreinopredito = TesteNN.propaga(conjTreino)

print("Ytreino predito inicial: ", Ytreinopredito)
T = pkt.Treinador(TesteNN)
#0.5 parece ser um bom passo
TesteNN = T.treinar(conjTreino, Ytreino, 0.01, 100, 0.0005)
Ytreinopredito = TesteNN.propaga(conjTreino)
print("Ytreino predito final: ", Ytreinopredito)

plt.plot(T.J, 'r-', linewidth=2.0)
plt.grid(1)
plt.show()

#############Testando no conjunto teste#############

YtestePredito = TesteNN.propaga(conjTeste)
print("Yteste real: ", Yteste)
print("Yteste predito final: ", YtestePredito)
print("erro final: ", T.J[-1])
YtesteErro = TesteNN.funcaoCusto(conjTeste,Yteste)
print("Erro de teste: ", YtesteErro)

#testando os gradientes
# grad = TesteNN.computaGradientes(dadosFechamento, Ytreino)
# numgrad = vlg.validaGradientes(TesteNN, dadosFechamento, Ytreino)
# print(grad)
# print("______")
# print(numgrad)

