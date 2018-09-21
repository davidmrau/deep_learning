import pickle


import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--filename', type = str, default = None,
                  help='filename')



FLAGS, unparsed = parser.parse_known_args()
acc, val_losses, train_losses = pickle.load(open(FLAGS.filename+".p", "rb" ) )



# plot accuracies and losses
plt.subplot(2, 1, 1)
plt.plot(acc[:][0],acc[:][1], 'o-')
#plt.title('Pytorch ConvNet')
#plt.ylim([0,80])
plt.ylabel('Accuracy (%)')

plt.subplot(2, 1, 2)
plt.plot(train_losses[:][0], train_losses[:][1])
plt.ylabel('Training Loss')

plt.subplot(3, 1, 3)
plt.plot(val_losses[:][0], val_losses[:][1])
plt.ylabel('Validation Loss')
#plt.ylim([0,100])
plt.xlabel('Epoch')
plt.savefig(FLAGS.filename+'.png')
