import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(all_true, all_pred, class_names):

    cm = confusion_matrix(all_true, all_pred)

    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Faster R-CNN")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


# Example usage
class_names = ["background", "deer", "elephant", "tiger", "bear", "leopard"]

plot_confusion_matrix(all_true, all_pred, class_names)