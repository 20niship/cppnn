import numpy as np 
import matplotlib.pyplot as plt

# W1 = np.loadtxt('W1.csv')
W1 = np.genfromtxt('W1.csv', delimiter=',')
W2 = np.genfromtxt('W2.csv', delimiter=',')

W1 = W1[:,:-1]

plt.figure(figsize=(10,5))
# for idx in range(W1.shape[1]):
for idx in range(10):
    w2a= W2[:, idx]

    print(w2a.shape)
    print(W1.shape)
    w = W1@w2a
    w= w.reshape(28, 28)
    plt.subplot(2,5, idx+1)
    plt.title("idx{}".format(idx))
    plt.imshow(w, cmap="Greys")
plt.show();


