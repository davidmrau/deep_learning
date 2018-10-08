import matplotlib.pyplot as plt
def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

import pickle

train_curve, test_curve = pickle.load(open('zdim_2/curves.p', 'rb'))
save_elbo_plot(train_curve, test_curve, 'curves.png')
