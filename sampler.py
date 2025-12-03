import numpy as np
import matplotlib.pyplot as plt

# Example: sampled data from a normal distribution
samples = -np.log(1-np.random.uniform(0, 1, 1000))



plt.figure()
plt.hist(samples, bins=30, density=True, alpha=0.6)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Histogram of Sampled Distribution")
plt.show()
