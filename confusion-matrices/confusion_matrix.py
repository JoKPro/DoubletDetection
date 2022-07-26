from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

matrix = np.array([[76, 6, 46],
                   [3, 2, 102],
                   [66, 0, 3175]])

class_names = ['heterotypic', 'homotypic', 'singlet']

fig, ax = plot_confusion_matrix(conf_mat=matrix,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                class_names=class_names)
plt.show()
