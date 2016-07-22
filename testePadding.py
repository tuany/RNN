import csv, os
import numpy as np

f = csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')
dadosFechamento = []
for linha in f:
	dadosFechamento.append(float(linha[4]))
dadosFechamento = dadosFechamento/np.amax(dadosFechamento, axis= 0)
dadosFechamento = np.reshape(dadosFechamento,(len(dadosFechamento),1))
#print(dadosFechamento)
delay = 5
aux = [[] for i in range(len(dadosFechamento+delay))]

for i in range(delay):
	