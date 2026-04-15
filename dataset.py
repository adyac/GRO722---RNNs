import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):

        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        # Lecture et chargement données comme liste [x1, x2]... [y1, y2] dans dictionnaire
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        # À compléter
        # création des dictionnaires pour encoder/décoder les symboles
        self.symb2int = dict()
        # reserver premiers indices aux symboles speciaux
        self.symb2int['word'] = {start_symbol: 0, stop_symbol: 1, pad_symbol: 2}

        # transformer chaque lettre en 1 entier des 26 lettres apres idx 2
        letters = 'abcdefghijklmnopqrstuvwxyz'
        for idx, char in enumerate(letters):
            self.symb2int['word'][char] = idx + 3

        # dictionnaire inverse pour retrouver lettres a partir des predictions en entiers
        self.int2symb = dict()
        self.int2symb['word'] = {}
        for char in self.symb2int['word']:
            idx = self.symb2int['word'][char]
            self.int2symb['word'][idx] = char

        # calculer tailles maximales des mots et des points dans l'écriture a la main
        # sert a trouver longueur fixe de toutes les séquences pour l'apprentissage en lots
        self.max_len = {
            'word': max(len(seq[0]) for seq in self.data) + 1,  #+1 pour <eos>
            'handwritten': max(seq[1].shape[1] for seq in self.data)  # nombre de points dans les coordonnées
        }

        # dictionnaires pour stocker les echantillons apres traitement
        self.padded_data = {'word': {}, 'handwritten': {}}

        # prétraitement de chaque échantillon
        for idx in range(len(self.data)):

            # récupérer le mot et la trajectoire
            word_str = list(self.data[idx][0])  # ex: ['h', 'e', 'l', 'l', 'o']
            coords = self.data[idx][1]  # ex: np.array([[x1, x2, ...], [y1, y2, ...]])

            # ajouter <eos> à la fin du mot pour que le modele sache s'arreter dans sa prediction
            word_str.append(self.stop_symbol)

            # ajout padding pour que séquences aient la même taille
            while len(word_str) < self.max_len['word']:
                word_str.append(self.pad_symbol)

            # normalisation des coordonnées entre 0 et 1 par dimension
            for dim in range(coords.shape[0]):
                min_val = coords[dim].min()
                max_val = coords[dim].max()
                coords[dim] = (coords[dim] - min_val) / (max_val - min_val + 1e-8)  # normalisation [0, 1]

            # padding à la fin de la trajectoire: le dernier point pour pas introduire faux points dans trajectoire

            last_point = coords[:, -1]  # forme (2,)
            while coords.shape[1] < self.max_len['handwritten']:
                coords = np.concatenate((coords, last_point.reshape(-1, 1)), axis=1)

            # stocker les données préparées
            self.padded_data['word'][idx] = word_str
            self.padded_data['handwritten'][idx] = coords

        self.dict_size = {'word': len(self.int2symb['word'])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # À compléter
        # récupérer les données et lettres padées et normalisées pour les envoyer au modele
        coords = self.padded_data['handwritten'][idx]
        word_letters = self.padded_data['word'][idx]

        # convertit chaque lettre en entier
        word_idx = []
        for letter in word_letters:
            idx = self.symb2int['word'][letter]
            word_idx.append(idx)

        # transposer coords pour correspondre a forme attendue par modèle (T,2)
        return torch.tensor(coords.T, dtype=torch.float32), torch.tensor(word_idx, dtype=torch.long)

    def visualisation(self, idx):
        # À compléter (optionel)
        coords_tensor, word_tensor = self[idx]

        # transformer tenseurs en tableaux
        coords = coords_tensor.numpy()
        word_idx = word_tensor.numpy()

        # reconvertit les indices en lettres à l'aide du dictionnaire inverse
        word_letters = []
        for idx in word_idx:
            letter = self.int2symb['word'][int(idx)]

            # ignorer les symboles spéciaux pour afficher que les lettres du mot
            if letter not in ['<sos>', '<eos>', '<pad>']:
                word_letters.append(letter)

        # tracer mouvement (x, y) selon temps
        plt.plot(coords[0], coords[1])  #x = coord 0, y = coord 1
        plt.title(''.join(word_letters))
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))