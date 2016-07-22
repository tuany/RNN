import csv, os
from datetime import datetime, timedelta
import operator, collections
import testeProsDados as ppd
import Neural_Network as nn

import matplotlib.pyplot as plt

vetorTabelas = []
vetorTabelas.append(list(csv.reader(open(os.getcwd()+'\index-data\MERV.csv'), delimiter=',')))
vetorTabelas.append(list(csv.reader(open(os.getcwd()+'\index-data\IBOVESPA.csv'), delimiter=',')))

prosDados = ppd.PreProcessData()

prosDados.setDados(vetorTabelas)

data = prosDados.getDados()

NN = nn.Neural_Network(1,1,3,0.000001)
x = NN.propaga(data[1])
#print(x)
#print(data[1])
data[1] = [val[1] for val in data[1].items()]
print(data[1])
#plt.plot(data[1])
plt.plot(x)
plt.show()
'''
for i in range(len(data)):
	data[i] = [val[1] for val in data[i].items()]

plt.plot(data[0])
plt.plot(data[1])
plt.show()

print("________")

print(data[1])'''