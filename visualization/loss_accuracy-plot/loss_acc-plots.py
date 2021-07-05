import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

# baselines, data contents and file names
models = ['CNN', 'RNN', 'RCNN','HAN','Transformer']
types  = ['loss','accuracy']
dataset= ['train','validation']
files  = ['run-train-tag-loss_loss_train.csv', 'run-validation-tag-accuracy.csv']

fig = plt.figure(figsize=(15,6))

# read and plot the train loss and the validation accuracy of baselines
for j,filetype in enumerate(files):
    # initialise the plotted data 
    x  = [0]
    y  = [0 for k in range(len(models))]
    
    ax  = plt.subplot2grid((1, 2), (0,j), colspan=1)
    # plot data
    for i,model in enumerate(models):
        # read data from files
        data = read_csv('./data/{}_{}'.format(model,filetype), header=None, skiprows=1,\
        delimiter=',',usecols=[1,2])
        x = data[1]
        y[i] = data[2]

        # plot the present model
        if j==0:
            ax.plot(x/1000,y[i])
            ax.set_xlabel('number of iterations(k)')
        else:
            ax.plot(x,y[i])
            ax.set_xlabel('number of epochs')
            ax.set_xticks(np.arange(0,19,2))
            ax.set_xticklabels(["$0$","$2$","$4$","$6$","$8$","$10$","$12$","$14$","$16$","$18$" ])
            ax.set_ylim(0.81,0.87)
        ax.set_ylabel('{} {}'.format(dataset[j],types[j]))
    ax.legend(models,loc=0)
plt.savefig('loss_accuracy.png')
plt.close()
