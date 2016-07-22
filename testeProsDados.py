import csv, os
from datetime import datetime, timedelta
import operator, collections

class PreProcessData():
	def __init__(self):
		self.vetorTabelas = collections.OrderedDict()
	#vão vir no objeto passado
	#g = csv.reader(open(os.getcwd()+'\index-data\MERV.csv'), delimiter=',')
	#f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
	def setDados(self, vetorTabelas):
		for i in vetorTabelas:
			self.vetorTabelas = vetorTabelas
	
	def getDados(self):
		dadosFechamento = collections.OrderedDict()
		for j, tabela in enumerate(self.vetorTabelas):
			dadosAux = collections.OrderedDict()
			for linha in tabela:
				dia = datetime.strptime(linha[0], '%Y-%m-%d').date()
				dadosAux[dia] = float(linha[4])
			amax = max(dadosAux.items(), key=operator.itemgetter(1))[1]
			amin = min(dadosAux.items(), key=operator.itemgetter(1))[1]
			dadosAux = {k: ((v - amin)/(amax-amin)) for k, v in dadosAux.items()}
			dadosOrdenados = sorted(dadosAux.items())
			dadosAux = collections.OrderedDict(dadosOrdenados)
			dadosFechamento[j] = dadosAux
		
		return dadosFechamento
		'''
		dtInicio = list(dadosFechamento[0].keys())[0]
		dtFim = list(dadosFechamento[0].keys())[-1]
		dt = dtInicio
		
		for i, tabela in enumerate(dadosFechamento):
			fechamentoAnterior = dadosFechamento[i][dtInicio]
			for days in range(int((dtFim - dtInicio).days)):
				dadosFechamento.setdefault(dt, fechamentoAnterior)
				fechamentoAnterior = dadosFechamento[i][dt]
				dt = dt + timedelta(days=1)
				
		############# Slices em conjunto treinamento/teste ####################
		dadosOrdenados = {sorted(dadosFechamento[i].items()) for i, tabela in dadosFechamento}
		print(dadosOrdenados)
		exit()
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
		return [conjTeste, conjTreino]'''