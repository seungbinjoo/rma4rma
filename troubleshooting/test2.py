import matplotlib.pyplot as plt
import numpy as np

# Generate data for plotting
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.figure()
plt.plot(x, y, label="Sine Wave")
plt.title("Simple Sine Wave Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)

# Display the plot
plt.show()