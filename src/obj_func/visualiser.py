import numpy as np
import matplotlib.pyplot as plt

from obj_func.simple import quad, almost_twin_peaks

import numpy as np
import matplotlib.pyplot as plt

def plot_population_on_objective(func, population, dim1, dim2):
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            temp_population = np.zeros((1, population.shape[1]))
            temp_population[:, dim1] = X[i, j]
            temp_population[:, dim2] = Y[i, j]
            Z[i, j] = func(temp_population)[0]

    plt.figure(figsize=(6, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='gray')

    plt.scatter(population[:, dim1], population[:, dim2], color='red', s=1)

    plt.title(f'Contour plot (Dimensions {dim1} and {dim2})')
    plt.xlabel(f'Dimension {dim1}')
    plt.ylabel(f'Dimension {dim2}')
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    population_size = 20
    dimensions = 5 
    population = np.random.rand(population_size, dimensions) * 20 - 10

    plot_population_on_objective(almost_twin_peaks, population, 0, 1)
