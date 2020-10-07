import numpy as np
import matplotlib.pyplot as plt

Y = np.load('noise_2.npy')/100
x = np.linspace(0, Y.shape[1]//10, Y.shape[1])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, Y[0], 'b.', label="training accuracy")
ax.plot(x, Y[1], 'r', markersize=2, label='testing accuracy')
plt.xlim(0, 110)
plt.ylim(0.7, 1)
ax.set_ylabel("accuracy", color='red')
ax.set_xlabel("epochs")
ax.tick_params(axis='y', colors='red')
ax.legend()
plt.show()
