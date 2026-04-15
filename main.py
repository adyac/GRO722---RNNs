# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from models import *
from dataset import *
from metrics import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = True           # Forcer a utiliser le cpu?
    training = False           # Entrainement?
    test = True            # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité initialement 1, choisi 42
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    batch_size = 64
    learning_rate = 0.01
    n_epochs = 300

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    # À compléter
    dataset = HandwrittenWords('data_trainval.p')


    # Séparation de l'ensemble de données (entraînement et validation)
    # À compléter
    n_train_samples = int(len(dataset) * 0.8)
    n_val_samples = len(dataset) - n_train_samples
    dataset_train, dataset_val = random_split(dataset, [n_train_samples, n_val_samples])

    # Instanciation des dataloaders
    # À compléter
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=n_workers)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=n_workers)


    # Instanciation du model
    # À compléter

    # + layer = améliorer performances
    # modele est assez complexe (birirectionnel avec attention), handwrittenWords est modeste en taille
    # 1 LSTM suffisant pour capter patterns utiles

    model = trajectory2seq(
        hidden_dim = 18,
        n_layers=1,
        symb2int=dataset.symb2int,
        int2symb=dataset.int2symb,
        dict_size=dataset.dict_size,
        device=device,
        maxlen=dataset.max_len
    )

    if training:
        # Fonction de coût et optimizateur
        # À compléter

        # Ameliore descente de gradient stochastique
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # on veut ignorer pad, on veut pas forcer modele a apprendre a predire pad, pour pas pénaliser modele
        criterion = nn.CrossEntropyLoss(ignore_index=2)

        print(f"Nombre total de paramètres du modèle: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        train_losses, val_losses = [], []
        train_dists, val_dists = [], []

        for epoch in range(1, n_epochs + 1):
            print('\nEpoch: %d' % epoch)
            # Entraînement
            # À compléter
            total_loss = 0
            total_dist = 0
            model.train()

            for inputs, targets in train_loader:
                # target forme (batch sz, seq len)
                # inputs forme (batch siz, seq len, vocab sz)
                inputs, targets = inputs.to(device).float(), targets.to(device)

                optimizer.zero_grad()
                # forward pass
                outputs, _, _ = model(inputs)

                # aplatir sequences pour cross entropy qui veut outputs size: (N,C) et target size : (N,)
                loss = criterion(outputs.view(-1, model.dict_size['word']), targets.view(-1))

                loss.backward()
                optimizer.step() # Mise a jour weights pour réduire loss
                total_loss += loss.item() #accum perte

                preds = torch.argmax(outputs, dim=-1).detach().cpu().tolist()
                targets_cpu = targets.cpu().tolist()

                # pour ch paire (pred, target) dans lot
                for p, t in zip(preds, targets_cpu):
                    # arreter pred au <eos> (idx = 1)
                    try:
                        p = p[:p.index(1)]
                    except ValueError:
                        pass

                    # arreter target au <eos> (idx = 1)
                    t = t[:t.index(1)]

                    # Mesure et accumule distance pour cette paire pred-target
                    total_dist += edit_distance(p, t)

            # stockage pour tracer les courbes
            avg_loss = total_loss / len(train_loader)
            avg_dist = total_dist / len(train_loader.dataset)
            train_losses.append(avg_loss)
            train_dists.append(avg_dist)

            # Validation
            # À compléter
            model.eval()
            val_loss = 0
            val_dist = 0

            val_preds_all, val_targets_all = [], []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device).float(), targets.to(device)
                    outputs, _, _ = model(inputs)
                    loss = criterion(outputs.view(-1, model.dict_size['word']), targets.view(-1))
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=-1).detach().cpu().tolist()
                    targets_cpu = targets.cpu().tolist()

                    for p, t in zip(preds, targets_cpu):
                        try:
                            p = p[:p.index(1)]
                        except ValueError:
                            pass
                        t = t[:t.index(1)]
                        val_dist += edit_distance(p, t)

                        pred_str = [dataset.int2symb['word'][idx] for idx in p]
                        true_str = [dataset.int2symb['word'][idx] for idx in t]
                        val_preds_all.append(pred_str)
                        val_targets_all.append(true_str)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_dist = val_dist / len(val_loader.dataset)
            val_losses.append(avg_val_loss)
            val_dists.append(avg_val_dist)

            # Display confusion matrix
            labels = list('abcdefghijklmnopqrstuvwxyz')
            #confusion_matrix(val_targets_all, val_preds_all, labels)

        if learning_curves:
            # visualization
            # À compléter

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label="Train")
            plt.plot(val_losses, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.title("Loss")

            plt.subplot(1, 2, 2)
            plt.plot(train_dists, label="Train")
            plt.plot(val_dists, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Edit Distance")
            plt.legend()
            plt.grid()
            plt.title("Edit Distance")

            plt.tight_layout()
            plt.pause(0.01)

            if learning_curves:
                plt.show()

        torch.save(model.state_dict(), "model.pt")

    if test:
        # Évaluation
        # À compléter
        model.load_state_dict(torch.load("model.pt"))
        model.eval()
        dataset_test = dataset

        # garantit longueur max du modèle = longueur max du test dataset, eviter erreurs de dimensions dans test
        model.maxlen['handwritten'] = dataset_test.max_len['handwritten']
        total_dist = 0
        preds_all, targets_all = [], []

        for i in range(len(dataset_test)):
        # prendre x, y de chaque mot dans dataset
            x, y = dataset_test[i]

            # ajout de la dimension batch et passage sur le bon device
            x_input = x.unsqueeze(0).to(device).float()
            y = y.to(device)

            # forward
            output, _, attn = model(x_input)

            # extraire pred a chaque temps (t) (argmax)
            pred = torch.argmax(output, dim=-1).cpu().squeeze().tolist()
            true = y.cpu().tolist()

            # Arreter pred et target à <eos> (index=1)
            try:
                pred = pred[:pred.index(1)]
            except ValueError:
                pass
            true = true[:true.index(1)]

            # calcul accum distance entre pred et target
            total_dist += edit_distance(pred, true)
            # transformer indices en lettres pour l'affichage
            pred_str = [dataset_test.int2symb['word'][idx] for idx in pred]
            true_str = [dataset_test.int2symb['word'][idx] for idx in true]
            # stocker pour matrice confusion
            preds_all.append(pred_str)
            targets_all.append(true_str)

            # Affichage de l'attention
            # À compléter (si nécessaire)

            # on veut juste 3 images
            if gen_test_images and i < 3:

                # debug print
                print(f"Gen graphique pour numéro : {i}")

                # coords doit etre (2, nombre_points) donc .T pour faciliter selection coord [0,:] et [1,:]
                coords = x.numpy().T
                num_points = coords.shape[1]  # 457
                # retirer dimension taille du batch: (taille batch, T, max_len) -> (T, max_len) pour manipul
                attn_map = attn.squeeze(0).cpu().detach().numpy()  # (457, max_len_word)

                # rendre un subplot verticalement espacé beau
                fig, axes = plt.subplots(len(pred_str), 1, figsize=(6, 2 * len(pred_str)))

                # gerer un seul caractère car matplotlib n'envoie pas de liste, mais faut transformer en liste
                if len(pred_str) == 1:
                    axes = [axes]

                # pour ch lettre prédite
                for j in range(len(pred_str)):

                    # extraction colonne attention pour chaque x, y
                    attn_j = attn_map[:, j]  # (457, )

                    # normalise entre [0,1] et inverser:  pour couleurs sur plot
                    attn_j = (attn_j - attn_j.min()) / (attn_j.max() - attn_j.min() + 1e-8)
                    attn_j = 1 - attn_j

                    # afficahe x,y coloré selon importance d'attention
                    axes[j].scatter(coords[0, :], coords[1, :], c=attn_j, cmap='gray')
                    axes[j].plot(coords[0, :], coords[1, :], c='lightgray')
                    axes[j].set_ylabel(pred_str[j])
                    axes[j].set_xticks([])
                    axes[j].set_yticks([])

                plt.suptitle(f"Target: {' '.join(true_str)}\nPrediction: {' '.join(pred_str)}")
                plt.tight_layout()
                plt.show()

        # calcul et affichage matrice confusion
        labels = list('abcdefghijklmnopqrstuvwxyz')
        confusion_matrix(targets_all, preds_all, labels)
