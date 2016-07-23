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
tamInput = 1
tamCamadaEsc = 2
tamCamadaSaida = 1
lambdaVal = 0.00001
timespan = 7 # janela de previsão (24h = 1, 48h = 2, 7d = 7 e 1m = 21)
index = "NIKKEI"
###########Leitura das tabelas##########
# Jogar isso num script de tratamento dos dados
nomeArquivo = "\index-data\%(index)s.csv" % locals()

g = csv.reader(open(os.getcwd()+nomeArquivo), delimiter=',')
f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
dadosFechamento = {}
entrada = {}
for linha in f:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	dadosFechamento[dia] = float(linha[4])
for linha in g:
	dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
	entrada[dia] = float(linha[4])

amax = max(dadosFechamento.items(), key=operator.itemgetter(1))[1]
amin = min(dadosFechamento.items(), key=operator.itemgetter(1))[1]

dadosFechamento = {k: ((v - amin)/(amax-amin)) for k, v in dadosFechamento.items()}

amaxM = max(entrada.items(), key=operator.itemgetter(1))[1]
aminM = min(entrada.items(), key=operator.itemgetter(1))[1]
entrada = {k: ((v - aminM)/(amaxM-aminM)) for k, v in entrada.items()}

dadosOrdenados = sorted(dadosFechamento.items())
entradaOrd = sorted(entrada.items())
dadosFechamento = collections.OrderedDict(dadosOrdenados)
entrada = collections.OrderedDict(entradaOrd)
dtInicio = list(entrada.keys())[0]
dtFim = list(entrada.keys())[-1]
dt = dtInicio
fechamentoAnterior = dadosFechamento[dtInicio]
fechamentoEntrada = list(entrada.items())[0][1]
for days in range(int((dtFim - dtInicio).days)):
	dadosFechamento.setdefault(dt, fechamentoAnterior)
	entrada.setdefault(dt, fechamentoEntrada)
	fechamentoAnterior = dadosFechamento[dt]
	fechamentoEntrada = entrada[dt]
	dt = dt + timedelta(days=1)
############# Slices em conjunto treinamento/teste ####################
dadosOrdenados = sorted(dadosFechamento.items())
entradaOrd = sorted(entrada.items())
slice = round(0.66 * len(entrada))

conjTreino = collections.OrderedDict(entradaOrd[:slice])
dtInicio = list(conjTreino.keys())[0] + timedelta(days=timespan)
dtFim = list(conjTreino.keys())[-1] + timedelta(days=timespan)

# faz o shift de 24h
Ytreino = {k: v for k, v in dadosFechamento.items() if k >= dtInicio and k <= dtFim}
Ytreino = collections.OrderedDict(sorted(Ytreino.items()))

conjTeste = collections.OrderedDict(entradaOrd[slice:])
dtInicio = list(conjTeste.keys())[0] + timedelta(days=timespan)
dtFim = list(conjTeste.keys())[-1] + timedelta(days=timespan)

# faz o shift de 24h
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

ax.plot(np.ravel(list(Ytreino.values())), linestyle='-', color='red', label='Ibovespa', linewidth=1) 
ax.plot(np.ravel(list(conjTreino.values())), linestyle=':', color='purple', label=index, linewidth=1) 
ax.plot(np.ravel(Ytreinopredito.T),'blue', label="Predito Final", linewidth=1) 
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.title("Ibovespa Real X Predito com Timespan %(timespan)s dia(s)" % locals())
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
