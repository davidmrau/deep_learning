import pickle


import matplotlib.pyplot as plt
accuracies = pickle.load(open( "accuracies.p", "rb" ) )
losses = pickle.load(open( "losses.p", "rb" ) )



# plot accuracies and losses
plt.subplot(2, 1, 1)
plt.plot(accuracies[:][0],accuracies[:][1], 'o-')
plt.title('Pytorch ConvNet')
plt.ylabel('Accuracy (%)')

plt.subplot(2, 1, 2)
plt.plot(losses[:][0], losses[:][1])
plt.xlabel('Step')
plt.ylabel('Pytorch ConvNet')

plt.savefig('pytorch_cnn.png')
