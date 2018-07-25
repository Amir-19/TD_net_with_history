import matplotlib.pyplot as plt
import numpy as np

errors = np.asarray(np.loadtxt('data/rmse.txt', dtype=float))
plt.plot(errors)
plt.show()
