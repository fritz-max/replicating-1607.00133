import numpy as np
import matplotlib.pyplot as plt


exp2 = (np.load("train_accuracies_noise2.npy")/100,
        np.load("test_accuracies_noise2.npy")/100)


exp4 = (np.load("train_accuracies_noise4.npy")[:500]/100,
        np.load("test_accuracies_noise4.npy")[:500]/100)

exp8 = (np.load("train_accuracies_noise8.npy")/100,
        np.load("test_accuracies_noise8.npy")/100)

experiments = [exp2, exp4, exp8]

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(14,4))

for ax, (y_train, y_test) in zip(reversed(axes), experiments):
    Y = np.ndarray((2, *y_train.shape))
    Y[0] = y_train
    Y[1] = y_test
    x = np.linspace(0, Y.shape[1]//10, Y.shape[1])
    ax.plot(x, Y[0], 'b.', markersize=1, label="training accuracy")
    ax.plot(x, Y[1], 'r', linewidth=0.5, label='testing accuracy')
    ax.set_ylabel("accuracy", color='red')
    ax.set_xlabel("epochs")
    ax.tick_params(axis='y', colors='red')
    ax.legend()
    ax.yaxis.set_tick_params(labelbottom=True)

plt.xlim(0, 110)
plt.ylim(0.65, 1)
plt.show()
