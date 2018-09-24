import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

y_srn = [100,100,100,82.2, 57.03, 38.28, 35.15,27.34, 25.78, 23.43, 24.21, 21.03, 20.31, 21.09 ,22.65, 20.31, 21.87]
y_lstm_0_01 = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
y_lstm = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 31.25, 23.09, 25, 26.56, 28.9, 23.43]
x_label = [5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,50]
x = np.arange(len(y_lstm))
plt.plot(x,y_srn,linestyle='-', marker='o', label='Vanilla RNN lr: 0.001')
plt.plot(x,y_lstm,linestyle='-', marker='o', color='g',label='LSTM lr: 0.001', alpha=0.40)
plt.plot(x,y_lstm_0_01,linestyle='-', marker='o', color='g', label='LSTM lr: 0.01')
plt.xticks(x, x_label)
plt.xlabel('Sequence length')
plt.legend(loc=7)
plt.ylabel('Accuracy (%)')

plt.show()
