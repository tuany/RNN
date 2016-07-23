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
tamInput = 3
tamCamadaEsc = 5
tamCamadaSaida = 1
lambdaVal = 0.00001
timespan = 21 # janela de previsão (24h = 1, 48h = 2, 7d = 7 e 1m = 21)
index1 = "IPC"
index2 = "EURONEXT-PARIS"
index3 = "HANGSENG"
###########Leitura das tabelas##########
nomeArquivo1 = "\index-data\%(index1)s.csv" % locals()
nomeArquivo2 = "\index-data\%(index2)s.csv" % locals()
nomeArquivo3 = "\index-data\%(index3)s.csv" % locals()
# Jogar isso num script de tratamento dos dados
g = csv.reader(open(os.getcwd()+nomeArquivo1), delimiter=',')
h = csv.reader(open(os.getcwd()+nomeArquivo2), delimiter=',')
i = csv.reader(open(os.getcwd()+nomeArquivo3), delimiter=',')
f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
dadosFechamento = {}
entrada1 = {}
entrada2 = {}
entrada3 = {}
for linha in f:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	dadosFechamento[dia] = float(linha[4])
for linha in g:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	entrada1[dia] = float(linha[4])
for linha in h:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	entrada2[dia] = float(linha[4])
for linha in i:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	entrada3[dia] = float(linha[4])

amax = max(dadosFechamento.items(), key=operator.itemgetter(1))[1]
amin = min(dadosFechamento.items(), key=operator.itemgetter(1))[1]

dadosFechamento = {k: ((v - amin)/(amax-amin)) for k, v in dadosFechamento.items()}

amax1 = max(entrada1.items(), key=operator.itemgetter(1))[1]
amin1 = min(entrada1.items(), key=operator.itemgetter(1))[1]
entrada1 = {k: ((v - amin1)/(amax1-amin1)) for k, v in entrada1.items()}

amax2 = max(entrada2.items(), key=operator.itemgetter(1))[1]
amin2 = min(entrada2.items(), key=operator.itemgetter(1))[1]
entrada2 = {k: ((v - amin2)/(amax2-amin2)) for k, v in entrada2.items()}

amax3 = max(entrada3.items(), key=operator.itemgetter(1))[1]
amin3 = min(entrada3.items(), key=operator.itemgetter(1))[1]
entrada3 = {k: ((v - amin3)/(amax3-amin3)) for k, v in entrada3.items()}

dadosOrdenados = sorted(dadosFechamento.items())
entrada1Ord = sorted(entrada1.items())
entrada2Ord = sorted(entrada2.items())
entrada3Ord = sorted(entrada3.items())
dadosFechamento = collections.OrderedDict(dadosOrdenados)
entrada1 = collections.OrderedDict(entrada1Ord)
entrada2 = collections.OrderedDict(entrada2Ord)
entrada3 = collections.OrderedDict(entrada3Ord)
dtInicio = list(entrada1.keys())[0]
dtFim = list(entrada1.keys())[-1]
dt = dtInicio
fechamentoAnterior = dadosFechamento[dtInicio]
fechamentoentrada1 = list(entrada1.items())[0][1]
fechamentoentrada2 = list(entrada2.items())[0][1]
fechamentoentrada3 = list(entrada3.items())[0][1]
entradas = {}
for days in range(int((dtFim - dtInicio).days)):
	dadosFechamento.setdefault(dt, fechamentoAnterior)
	entrada1.setdefault(dt, fechamentoentrada1)
	entrada2.setdefault(dt, fechamentoentrada2)
	entrada3.setdefault(dt, fechamentoentrada3)
	entradas.setdefault(dt, [fechamentoentrada1, fechamentoentrada2, fechamentoentrada3])
	fechamentoAnterior = dadosFechamento[dt]
	fechamentoentrada1 = entrada1[dt]
	fechamentoentrada2 = entrada2[dt]
	fechamentoentrada3 = entrada3[dt]
	dt = dt + timedelta(days=1)
############# Slices em conjunto treinamento/teste ####################
entrada1Ord = sorted(entrada1.items())
entrada2Ord = sorted(entrada2.items())
entrada3Ord = sorted(entrada3.items())
dadosOrdenados = sorted(dadosFechamento.items())
entradasOrd = sorted(entradas.items())
slice = round(0.66 * len(entradas))

conjTreino = collections.OrderedDict(entradasOrd[:slice])
entrada1Treino = collections.OrderedDict(entrada1Ord[:slice])
entrada2Treino = collections.OrderedDict(entrada2Ord[:slice])
entrada3Treino = collections.OrderedDict(entrada3Ord[:slice])
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
TesteNN = T.treinar(conjTreino, Ytreino, 0.1, 10000, 0.00005)
Ytreinopredito = TesteNN.propaga(conjTreino)

# plt.plot(T.J, 'r-', linewidth=2.0)
# plt.grid(1)
# plt.show()

#############Testando no conjunto teste#############
YtestePredito = TesteNN.propaga(conjTeste)



fig = plt.figure()
ax = plt.subplot(111)

ax.plot(np.ravel(list(Ytreino.values())), linestyle='--', color='red', label='Ibovespa', linewidth=1.5) 
ax.plot(np.ravel(list(entrada1Treino.values())), linestyle=':', color='purple', label=index1, linewidth=2) 
ax.plot(np.ravel(list(entrada2Treino.values())), linestyle=':', color='green', label=index2, linewidth=2)
ax.plot(np.ravel(list(entrada3Treino.values())), linestyle=':', color='yellow', label=index3, linewidth=2) 
ax.plot(np.ravel(Ytreinopredito.T), linestyle='-', color='blue', label="Predito Final", linewidth=1) 
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.title("Ibovespa Real X Predito com Timespan %(timespan)s dia(s)" % locals())
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()