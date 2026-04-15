# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        # Definition des couches
        # Couches pour rnn
        # À compléter
        # encodage
        self.encoder = nn.LSTM(input_size=2, hidden_size=hidden_dim, num_layers=n_layers,
                               batch_first=True, bidirectional=True)

        # embedding = indices des lettres en vecteurs
        self.embedding = nn.Embedding(dict_size['word'], hidden_dim)

        # décodage : prédit une lettre à chaque étape
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers,
                               batch_first=True, bidirectional=False)

        # attention pour se concentrer sur les bonnes parties de l’entrée
        self.hidden2query = nn.Linear(hidden_dim, 2 * hidden_dim) # prepare le vecteur pour produit scalaire

        # combine out décodeur et attention
        self.attention_combine = nn.Linear(3 * hidden_dim, hidden_dim)

        # lineaire pour ramener a la bonne taille de dict pour predire
        self.fc = nn.Linear(hidden_dim, dict_size['word'])

    def forward(self, x):
        batch_size = x.shape[0]
        max_len_word = self.maxlen['word']
        max_len_traj = self.maxlen['handwritten']

        # encodeur out et hidden
        encoder_outs, (h, c) = self.encoder(x)

        if self.num_directions == 2:
            h = h[0:h.size(0):2] + h[1:h.size(0):2]
            c = c[0:c.size(0):2] + c[1:c.size(0):2]

        # init vecteur in du décodeur avec zéro debut (<sos>)
        vec_in = torch.full((batch_size, 1), self.symb2int['word']['<sos>'], dtype=torch.long).to(self.device)

        # init out
        output_seq = torch.zeros((batch_size, max_len_word, self.dict_size['word'])).to(self.device)

        # init weights attention
        attention_weights = torch.zeros((batch_size, max_len_traj, max_len_word)).to(self.device)

        # boucle décodeur

        hidden = (h, c)
        for t in range(max_len_word):
            embedded = self.embedding(vec_in)

            # décodeur pour prédire next lettre
            decoder_out, hidden = self.decoder(embedded, hidden)

            # attention
            query = self.hidden2query(decoder_out)  # transformer pour produit scalaire
            scores = torch.bmm(encoder_outs, query.transpose(1, 2))  # produit scalaire
            attn_weights = torch.softmax(scores, dim=1)
            context = torch.sum(attn_weights * encoder_outs, dim=1, keepdim=True)

            # concat contexte et out du décodeur
            combined = torch.cat((decoder_out, context), dim=2)
            combined = self.attention_combine(combined)

            # pred finale
            out = self.fc(combined)
            output_seq[:, t, :] = out[:, 0, :]

            # choisir la lettre suivante à prédire
            vec_in = torch.argmax(out, dim=2)

            # stock poids d'attention
            attention_weights[:, :, t] = attn_weights[:, :, 0]

        return output_seq, hidden, attention_weights