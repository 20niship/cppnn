import numpy as np 
import matplotlib.pyplot as plt

res = np.genfromtxt('result.csv', delimiter=',')

losstrain_n = res[0]
acctrain_n = res[1]
acctest_n= res[2]

losstrain = res[3]
acctrain = res[4]
acctest= res[5]

hidden= res[6]

plt.subplot(1,2,1)
plt.title("train loss")
plt.plot(hidden, losstrain, 'r--')
plt.plot(hidden, losstrain_n)

plt.subplot(1,2,2)
plt.title("acc")
plt.plot(hidden, acctest, 'r--', label="test")
plt.plot(hidden, acctrain, 'r--', label="train")

plt.plot(hidden, acctest_n, label="test(noise)")
plt.plot(hidden, acctrain_n, label="train(noise)")

plt.legend()
plt.show()


