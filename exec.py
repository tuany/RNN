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
lambdaVal = 0.00001
###########Leitura das tabelas##########
# Jogar isso num script de tratamento dos dados
g = csv.reader(open(os.getcwd()+'\index-data\MERV.csv'), delimiter=',')
f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
dadosFechamento = {}
merv = {}
for linha in f:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	dadosFechamento[dia] = float(linha[4])
for linha in g:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	merv[dia] = float(linha[4])

amax = max(dadosFechamento.items(), key=operator.itemgetter(1))[1]
dadosFechamento = {k: (v / amax) for k, v in dadosFechamento.items()}

amax = max(merv.items(), key=operator.itemgetter(1))[1]
merv = {k: (v / amax) for k, v in merv.items()}

dadosOrdenados = sorted(dadosFechamento.items())
mervOrd = sorted(merv.items())
dadosFechamento = collections.OrderedDict(dadosOrdenados)
merv = collections.OrderedDict(mervOrd)
dtInicio = list(merv.keys())[0]
dtFim = list(merv.keys())[-1]
dt = dtInicio
fechamentoAnterior = dadosFechamento[dtInicio]
fechamentoMerv = merv[dtInicio]
for days in range(int((dtFim - dtInicio).days)):
	dadosFechamento.setdefault(dt, fechamentoAnterior)
	merv.setdefault(dt, fechamentoMerv)
	fechamentoAnterior = dadosFechamento[dt]
	fechamentoMerv = merv[dt]
	dt = dt + timedelta(days=1)
############# Slices em conjunto treinamento/teste ####################
dadosOrdenados = sorted(dadosFechamento.items())
mervOrd = sorted(merv.items())
slice = round(0.66 * len(merv))

conjTreino = collections.OrderedDict(mervOrd[:slice])
dtInicio = list(conjTreino.keys())[0] + timedelta(days=21)
dtFim = list(conjTreino.keys())[-1] + timedelta(days=21)

# faz o shift de 24h
Ytreino = {k: v for k, v in dadosFechamento.items() if k >= dtInicio and k <= dtFim}
Ytreino = collections.OrderedDict(sorted(Ytreino.items()))

conjTeste = collections.OrderedDict(mervOrd[slice:])
dtInicio = list(conjTeste.keys())[0] + timedelta(days=21)
dtFim = list(conjTeste.keys())[-1] + timedelta(days=21)

# faz o shift de 24h
Yteste = {k: v for k, v in dadosFechamento.items() if k >= dtInicio and k <= dtFim}
Yteste = collections.OrderedDict(sorted(Yteste.items()))

#################################################################
TesteNN = NN.Neural_Network(tamInput, tamCamadaSaida, tamCamadaEsc, lambdaVal)
preditoInicial = TesteNN.propaga(conjTreino)

T = pkt.Treinador(TesteNN)
#0.5 parece ser um bom passo
TesteNN = T.treinar(conjTreino, Ytreino, 0.56, 100, 0.00005)
Ytreinopredito = TesteNN.propaga(conjTreino)

# plt.plot(T.J, 'r-', linewidth=2.0)
# plt.grid(1)
# plt.show()

#############Testando no conjunto teste#############
YtestePredito = TesteNN.propaga(conjTeste)

plt.axes(xlim=(0, 24))

plt.plot(np.ravel(list(Ytreino.values())),'red', label='Ibovespa', linewidth=2) 
plt.plot(np.ravel(list(conjTreino.values())),'purple', label='Merval', linewidth=2) 
plt.plot(np.ravel(preditoInicial),'green', label='Predito Inicial', linewidth=2)
plt.plot(np.ravel(Ytreinopredito.T),'blue', label="Predito Final", linewidth=2) 
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=2, mode="expand", borderaxespad=0., prop={'size':10})
plt.show()

# print("Yteste real: ", Yteste)
# print("Yteste predito final: ", YtestePredito)
# print("erro final: ", T.J[-1])
# YtesteErro = TesteNN.funcaoCusto(conjTeste,Yteste)
# print("Erro de teste: ", YtesteErro)

#testando os gradientes
# grad = TesteNN.computaGradientes(dadosFechamento, Ytreino)
# numgrad = vlg.validaGradientes(TesteNN, dadosFechamento, Ytreino)
# print(grad)
# print("______")
# print(numgrad)

