import pickle


import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--filename', type = str, default = None,
                  help='filename')



FLAGS, unparsed = parser.parse_known_args()
accuracies = pickle.load(open( "accuracies"+FLAGS.filename+".p", "rb" ) )
losses = pickle.load(open( "losses"+FLAGS.filename+".p", "rb" ) )



# plot accuracies and losses
plt.subplot(2, 1, 1)
plt.plot(accuracies[:][0],accuracies[:][1], 'o-')
#plt.title('Pytorch ConvNet')
plt.ylabel('Accuracy (%)')

plt.subplot(2, 1, 2)
plt.plot(losses[:][0], losses[:][1])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig(FLAGS.filename+'.png')
