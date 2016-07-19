import numpy as np
import Neural_Network as NN
import matplotlib.pyplot as plt
import PokeTrainer as pkt
import csv
import os
from datetime import date, datetime, timedelta
import collections
import operator

###########Parametros############
tamOutput = 1
tamInput = 1
tamCamadaEsc = 3
tamCamadaSaida = 1
###########Leitura das tabelas##########
# Jogar isso num script de tratamento dos dados
f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
dadosFechamento = {}
for linha in f:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	dadosFechamento[dia] = float(linha[4])

amax = max(dadosFechamento.items(), key=operator.itemgetter(1))[1]
print(amax)
dadosFechamento = {k: (v / amax) for k, v in dadosFechamento.items()}
dadosOrdenados = sorted(dadosFechamento.items())
dadosFechamento = collections.OrderedDict(dadosOrdenados)
dtInicio = list(dadosFechamento.keys())[0]
dtFim = list(dadosFechamento.keys())[-1]
dt = dtInicio
fechamentoAnterior = dadosFechamento[dtInicio]
for days in range(int((dtFim - dtInicio).days)):
	dadosFechamento.setdefault(dt, fechamentoAnterior)
	fechamentoAnterior = dadosFechamento[dt]
	dt = dt + timedelta(days=1)
############# Slices em conjunto treinamento/teste ####################
dadosOrdenados = sorted(dadosFechamento.items())
slice = round(0.66 * len(dadosFechamento))

conjTreino = collections.OrderedDict(dadosOrdenados[:slice])
dtInicio = list(conjTreino.keys())[0] + timedelta(days=1)
dtFim = list(conjTreino.keys())[-1] + timedelta(days=1)

# faz o shift de 24h
Ytreino = {k: v for k, v in dadosFechamento.items() if k >= dtInicio and k <= dtFim}
Ytreino = collections.OrderedDict(sorted(Ytreino.items()))

conjTeste = collections.OrderedDict(dadosOrdenados[slice:])
dtInicio = list(conjTeste.keys())[0] + timedelta(days=1)
dtFim = list(conjTeste.keys())[-1] + timedelta(days=1)

# faz o shift de 24h
Yteste = {k: v for k, v in dadosFechamento.items() if k >= dtInicio and k <= dtFim}
Yteste = collections.OrderedDict(sorted(Yteste.items()))

#################################################################
TesteNN = NN.Neural_Network(tamInput, tamCamadaSaida, tamCamadaEsc)
Ytreinopredito = TesteNN.propaga(conjTreino)

print("Ytreino predito inicial: ", Ytreinopredito)
T = pkt.Treinador(TesteNN)
#0.5 parece ser um bom passo
TesteNN = T.treinar(conjTreino, Ytreino, 0.98, 10, 0.00005)
Ytreinopredito = TesteNN.propaga(conjTreino)
print("Ytreino predito final: ", Ytreinopredito)

# plt.plot(T.J, 'r-', linewidth=2.0)
# plt.grid(1)
# plt.show()

#############Testando no conjunto teste#############
YtestePredito = TesteNN.propaga(conjTeste)

plt.axes(xlim=(0, 24))
plt.plot(np.ravel(list(Ytreino.values())),'r', label='Y', linewidth=2) 
plt.plot(np.ravel(Ytreinopredito.T),'b', label="Predito Treino", linewidth=2) 
plt.plot(np.ravel(YtestePredito.T),'g', label="Predito Teste", linewidth=2) 
plt.legend()
plt.show()

#testando os gradientes
# grad = TesteNN.computaGradientes(dadosFechamento, Ytreino)
# numgrad = vlg.validaGradientes(TesteNN, dadosFechamento, Ytreino)
# print(grad)
# print("______")
# print(numgrad)

