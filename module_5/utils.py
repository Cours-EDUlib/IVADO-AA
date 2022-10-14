import torch
from matplotlib import pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def frontieres_de_decision(X, y, reseau, couleurs_classes, cmap, titre):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    with torch.no_grad():
        Z = reseau(
            torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.FloatTensor)
        ).detach()
        if Z.size(1) > 1:
            Z = torch.argmax(Z, dim=1)
        Z = Z.numpy()

    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=cmap, shading="auto")

    for i in range(len(couleurs_classes)):
        plt.scatter(
            X[y == i, 0],
            X[y == i, 1],
            c=couleurs_classes[i],
            label=f"Classe {i}",
            edgecolors="k",
        )

    plt.title(titre)

    plt.legend(loc="upper right")


class JeuDeDonnees(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def entrainer_reseau(
    reseau,
    donnees_entrainement,
    donnes_validation,
    cycles,
    fct_perte,
    optimiseur,
    mini_batch_taille=None,
):
    train_loader = prep_dataloader(donnees_entrainement, mini_batch_taille)
    valid_loader = prep_dataloader(donnes_validation, mini_batch_taille)

    _entrainer_reseau(reseau, train_loader, valid_loader, cycles, fct_perte, optimiseur)


def prep_dataloader(donnees, mini_batch_taille):
    if isinstance(donnees, DataLoader):
        return donnees

    elif isinstance(donnees, Dataset):
        return DataLoader(donnees, batch_size=mini_batch_taille)

    elif list(map(type, donnees)) == [np.ndarray, np.ndarray]:
        if mini_batch_taille is None:
            mini_batch_taille = 1
        return DataLoader(
            JeuDeDonnees(donnees[0], donnees[1]), batch_size=mini_batch_taille
        )

    else:
        raise ValueError(
            "The training and validation data must either be a dataloder or a tuple of numpy arrays"
        )


def _entrainer_reseau(
    reseau, train_loader, valid_loader, cycles, fct_perte, optimiseur
):
    for i in range(cycles):
        perte_entrainement = 0
        exactitude_entrainement = 0

        reseau.train()
        for mini_batch in train_loader:
            optimiseur.zero_grad()

            echantillons = mini_batch[0]
            verite = mini_batch[1]

            predictions = reseau(echantillons)

            valeur_perte = fct_perte(predictions, verite)

            valeur_perte.backward()

            optimiseur.step()

            perte_entrainement += valeur_perte
            exactitude_entrainement += (
                verite == torch.argmax(predictions, dim=1)
            ).float().sum() / verite.size(0)

        perte_entrainement /= len(train_loader)
        exactitude_entrainement /= len(train_loader)

        reseau.eval()
        with torch.no_grad():
            perte_valid = 0
            exactitude_valid = 0
            for mini_batch in valid_loader:
                echantillons_valid = mini_batch[0]
                predictions_valid = reseau(echantillons_valid)

                verite_valid = mini_batch[1]

                perte_valid += fct_perte(predictions_valid, verite_valid)

                exactitude_valid += (
                    verite_valid == torch.argmax(predictions_valid, dim=1)
                ).float().sum() / verite_valid.size(0)

            perte_valid /= len(valid_loader)
            exactitude_valid /= len(valid_loader)

        print(f"Cycle {i+1}:")
        print(
            f"Entrainement: Perte moyenne {perte_entrainement} -- Exactitude moyenne: {exactitude_entrainement}"
        )
        print(
            f"Validation: Perte moyenne {perte_valid} -- Exactitude moyenne: {exactitude_valid}\n"
        )
