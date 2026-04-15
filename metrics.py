# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np
import matplotlib.pyplot as plt


def edit_distance(x, y):
    # Calcul de la distance d'édition

    # À compléter
    # matrice de zero
    d = np.zeros((len(x) + 1, len(y) + 1))

    # si une matrice vide, coût = longueur de l’autre
    for i in range(len(x) + 1):
        d[i, 0] = i
    for j in range(len(y) + 1):
        d[0, j] = j

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:   # même caractère = pas de coût
                substitution = d[i - 1, j - 1]
            else:  # substitution avec coût +1
                substitution = d[i - 1, j - 1] + 1

            # coût minimum entre suppression, insertion, ou substitution
            d[i, j] = min(
                substitution,
                d[i - 1, j] + 1,  # suppression
                d[i, j - 1] + 1  # insertion
            )

    return d[len(x), len(y)]


def confusion_matrix(true, pred, labels, ignore=[]):
    # Calcul de la matrice de confusion

    # À compléter
    n_classes = len(labels) # letters possibles
    matrix = np.zeros((n_classes, n_classes), dtype=int) #26x26


    for i in range(len(true)):
        true_word = true[i]
        pred_word = pred[i]

        min_len = min(len(true_word), len(pred_word))  # comparer avec la longeur

        for t_char, p_char in zip(true_word[:min_len], pred_word[:min_len]): # même position dans la liste
            if t_char in labels and p_char in labels:
                row = labels.index(t_char)
                col = labels.index(p_char)
                matrix[row, col] += 1 # incrémente la bonne case

    # affichage
    # Affichage de la matrice
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues')

    # Étiquettes des axes
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Affichage des nombres dans les cases
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vrai")
    fig.tight_layout()
    plt.colorbar(im)
    plt.grid(False)
    plt.show()