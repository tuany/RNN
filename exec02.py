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
tamInput = 2
tamCamadaEsc = 3
tamCamadaSaida = 1
lambdaVal = 0.00001
timespan = 21 # janela de previsão (24h = 1, 48h = 2, 7d = 7 e 1m = 21)
###########Leitura das tabelas##########
# Jogar isso num script de tratamento dos dados
g = csv.reader(open(os.getcwd()+'\index-data\MERV.csv'), delimiter=',')
h = csv.reader(open(os.getcwd()+'\index-data\IPSA.csv'), delimiter=',')
f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
dadosFechamento = {}
merv = {}
ipsa = {}
for linha in f:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	dadosFechamento[dia] = float(linha[4])
for linha in g:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	merv[dia] = float(linha[4])
for linha in h:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	ipsa[dia] = float(linha[4])

amax = max(dadosFechamento.items(), key=operator.itemgetter(1))[1]
amin = min(dadosFechamento.items(), key=operator.itemgetter(1))[1]

dadosFechamento = {k: ((v - amin)/(amax-amin)) for k, v in dadosFechamento.items()}

amaxM = max(merv.items(), key=operator.itemgetter(1))[1]
aminM = min(merv.items(), key=operator.itemgetter(1))[1]
merv = {k: ((v - aminM)/(amaxM-aminM)) for k, v in merv.items()}

amaxI = max(ipsa.items(), key=operator.itemgetter(1))[1]
aminI = min(ipsa.items(), key=operator.itemgetter(1))[1]
ipsa = {k: ((v - aminI)/(amaxI-aminI)) for k, v in ipsa.items()}

dadosOrdenados = sorted(dadosFechamento.items())
mervOrd = sorted(merv.items())
ipsaOrd = sorted(ipsa.items())
dadosFechamento = collections.OrderedDict(dadosOrdenados)
merv = collections.OrderedDict(mervOrd)
ipsa = collections.OrderedDict(ipsaOrd)
dtInicio = list(merv.keys())[0]
dtFim = list(merv.keys())[-1]
dt = dtInicio
fechamentoAnterior = dadosFechamento[dtInicio]
fechamentoMerv = merv[dtInicio]
fechamentoIpsa = ipsa[dtInicio]
entradas = {}
for days in range(int((dtFim - dtInicio).days)):
	dadosFechamento.setdefault(dt, fechamentoAnterior)
	merv.setdefault(dt, fechamentoMerv)
	ipsa.setdefault(dt, fechamentoIpsa)
	entradas.setdefault(dt, [fechamentoMerv, fechamentoIpsa])
	fechamentoAnterior = dadosFechamento[dt]
	fechamentoMerv = merv[dt]
	fechamentoIpsa = ipsa[dt]
	dt = dt + timedelta(days=1)
############# Slices em conjunto treinamento/teste ####################
mervOrd = sorted(merv.items())
ipsaOrd = sorted(ipsa.items())
dadosOrdenados = sorted(dadosFechamento.items())
entradasOrd = sorted(entradas.items())
slice = round(0.66 * len(entradas))

conjTreino = collections.OrderedDict(entradasOrd[:slice])
mervalTreino = collections.OrderedDict(mervOrd[:slice])
ipsaTreino = collections.OrderedDict(ipsaOrd[:slice])
dtInicio = list(conjTreino.keys())[0] + timedelta(days=timespan)
dtFim = list(conjTreino.keys())[-1] + timedelta(days=timespan)

# faz o shift de timespan
Ytreino = {k: v for k, v in dadosFechamento.items() if k >= dtInicio and k <= dtFim}
Ytreino = collections.OrderedDict(sorted(Ytreino.items()))

conjTeste = collections.OrderedDict(entradasOrd[slice:])
dtInicio = list(conjTeste.keys())[0] + timedelta(days=timespan)
dtFim = list(conjTeste.keys())[-1] + timedelta(days=timespan)

# faz o shift de timespan
Yteste = {k: v for k, v in dadosFechamento.items() if k >= dtInicio and k <= dtFim}
Yteste = collections.OrderedDict(sorted(Yteste.items()))

#################################################################
TesteNN = NN.Neural_Network(tamInput, tamCamadaSaida, tamCamadaEsc, lambdaVal)
preditoInicial = TesteNN.propaga(conjTreino)

T = pkt.Treinador(TesteNN)
#0.5 parece ser um bom passo
TesteNN = T.treinar(conjTreino, Ytreino, 0.1, 1000, 0.00005)
Ytreinopredito = TesteNN.propaga(conjTreino)

# plt.plot(T.J, 'r-', linewidth=2.0)
# plt.grid(1)
# plt.show()

#############Testando no conjunto teste#############
YtestePredito = TesteNN.propaga(conjTeste)



fig = plt.figure()
ax = plt.subplot(111)

ax.plot(np.ravel(list(Ytreino.values())), linestyle='--', color='red', label='Ibovespa', linewidth=2) 
ax.plot(np.ravel(list(mervalTreino.values())), linestyle=':', color='purple', label='Merval', linewidth=2) 
ax.plot(np.ravel(list(ipsaTreino.values())), linestyle=':', color='green', label='IPSA', linewidth=2) 
ax.plot(np.ravel(Ytreinopredito.T),'blue', label="Predito Final", linewidth=2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.title("Ibovespa Real X Predito com Timespan %(timespan)s dia(s)" % locals())
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()