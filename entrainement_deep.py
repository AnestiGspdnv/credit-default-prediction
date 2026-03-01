import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)
import optuna
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pretraitement import lancer_pretraitement

# Focal Loss
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        probs = torch.sigmoid(logits)

        # probabilité pour la vraie classe
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # eviter log(0)
        p_t = torch.clamp(p_t, min=1e-7, max=1 - 1e-7)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        facteur_focal = (1 - p_t) ** self.gamma

    # Focal Loss = -α *(1-p_t)^γ *log(p_t)
        loss = -alpha_t * facteur_focal * torch.log(p_t)

        return loss.mean()

    def __repr__(self):
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma})"

# Réseau de Neurones
class ReseauCredit(nn.Module):

    def __init__(self, nb_features, couches=[128, 64, 32, 16], dropout=0.3):
        super().__init__()

        # faire des couches cachées automat
        self.blocs = nn.ModuleList()
        dim_precedente = nb_features

        for dim in couches:
            bloc = nn.Sequential(
                nn.Linear(dim_precedente, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.blocs.append(bloc)
            dim_precedente = dim

        # relie la couche 1 à 2
        if len(couches) >= 2:
            self.skip = nn.Linear(couches[0], couches[1])
            self.utiliser_skip = True
        else:
            self.utiliser_skip = False

        # sortie: 1 seul neurone
        self.sortie = nn.Linear(couches[-1], 1)

    def forward(self, x):
        for i, bloc in enumerate(self.blocs):
            x_avant = x
            x = bloc(x)

            # Skip connection après la couche 2
            if i == 1 and self.utiliser_skip:
                x = x + self.skip(x_avant)

        logits = self.sortie(x)
        return logits

# fonctions d'Entraînement
def entrainer_reseau(model, criterion, X_train, y_train, X_val, y_val, lr=0.001, epochs=100, batch_size=256, patience=15):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # convertir en tenseur Pytorch
    X_tr = torch.FloatTensor(np.array(X_train)).to(device)
    y_tr = torch.FloatTensor(np.array(y_train)).to(device)
    X_v = torch.FloatTensor(np.array(X_val)).to(device)
    y_v = torch.FloatTensor(np.array(y_val)).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimiseur Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    historique = {"train_loss": [], "val_loss": []}
    meilleure_val_loss = float("inf")
    meilleur_state = None
    compteur_patience = 0

    for epoch in range(epochs):
        model.train()
        losses_epoch = []

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses_epoch.append(loss.item())

        train_loss = np.mean(losses_epoch)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v).squeeze()
            val_loss = criterion(val_logits, y_v).item()

        scheduler.step(val_loss)
        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} "
                  f"Train: {train_loss:.4f} Val: {val_loss:.4f}")

        if val_loss < meilleure_val_loss:
            meilleure_val_loss = val_loss
            meilleur_state = {k: v.clone() for k, v in model.state_dict().items()}
            compteur_patience = 0
        else:
            compteur_patience += 1
            if compteur_patience >= patience:
                print(f"Arrêt anticipé à l'epoch {epoch+1}")
                break

    if meilleur_state:
        model.load_state_dict(meilleur_state)
        print(f"Meilleur modele restauré (val_loss={meilleure_val_loss:.4f})")

    return historique

def predire_proba(model, X):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    X_t = torch.FloatTensor(np.array(X)).to(device)
    with torch.no_grad():
        logits = model(X_t).squeeze()
        proba = torch.sigmoid(logits).cpu().numpy()
    return proba

# principal
def entrainer_deep_learning(X_train=None, X_val=None, X_test=None,y_train=None, y_val=None, y_test=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device}")

    if X_train is None:
        X_train, X_val, X_test, y_train, y_val, y_test = lancer_pretraitement()

    nb_features = X_train.shape[1]
    print(f"Nm de features : {nb_features}")

    # Partie 1 : Optimisation des hyperparamètres Optuna
    print("\n1. Optimisation Optuna")
    def objectif_nn(trial):
        n_couches = trial.suggest_int("n_couches", 2, 4)
        dims = []
        for i in range(n_couches):
            dims.append(trial.suggest_categorical(f"dim_{i}", [32, 64, 128, 256]))

        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        alpha = trial.suggest_float("alpha", 0.5, 0.9)
        gamma = trial.suggest_float("gamma", 0.5, 3.0)

        model = ReseauCredit(nb_features, couches=dims, dropout=dropout)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)

        # entraînement rapide pour l'optimisation
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        X_tr = torch.FloatTensor(np.array(X_train))
        y_tr = torch.FloatTensor(np.array(y_train))
        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(30):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx).squeeze(), by)
                loss.backward()
                optimizer.step()

        # evaluer sur la valid
        proba = predire_proba(model, X_val)
        try:
            return roc_auc_score(y_val, proba)
        except Exception:
            return 0.5

    study = optuna.create_study(direction="maximize")
    study.optimize(objectif_nn, n_trials=15)

    print(f"Meilleur AUC-ROC : {study.best_value:.4f}")
    print(f"Meilleurs paramètres: {study.best_params}")


    # Partie 2:
    print("\n2. Entraînement Final")

    bp = study.best_params
    n_couches = bp["n_couches"]
    dims = [bp[f"dim_{i}"] for i in range(n_couches)]

    print(f"Architecture: {dims}")
    print(f"Dropout: {bp['dropout']:.3f}")
    print(f"Learning rate: {bp['lr']:.6f}")
    print(f"Focal Loss: α ={bp['alpha']:.3f},γ={bp['gamma']:.3f}")

    model_final = ReseauCredit(nb_features, couches=dims, dropout=bp["dropout"])
    criterion_final = FocalLoss(alpha=bp["alpha"], gamma=bp["gamma"])

    print(f"\nCritère utilisé : {criterion_final}")

    historique = entrainer_reseau(
        model_final, criterion_final,
        X_train, y_train, X_val, y_val,
        lr=bp["lr"], epochs=150,
        batch_size=bp["batch_size"], patience=20
    )

    # Partie 3
    print("\n3. Courbes d'apprentissage")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(historique["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(historique["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Focal Loss")
    ax.set_title("Courbes d'apprentissage - Réseau de Neurones")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    #os.makedirs("resultats", exist_ok=True)
    #plt.savefig("resultats/courbes_apprentissage.png", dpi=150, bbox_inches="tight")
    #print("Sauvegardé : resultats/courbes_apprentissage.png")
    #plt.close()

    # Analyser les courbes
    gap = historique["val_loss"][-1] - historique["train_loss"][-1]
    print(f"Train loss finale: {historique['train_loss'][-1]:.4f}")
    print(f"Val loss finale: {historique['val_loss'][-1]:.4f}")
    print(f"Écart (overfitting): {gap:.4f}")

    if gap > 0.1:
        print("Un peu de surapprentissage, mais le early stopping l'a limité")
    else:
        print("Pas de surapprentissage visible")

    # Partie 4 :
    print("\n4. Évaluation sur le Test set")

    y_proba = predire_proba(model_final, X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"recall : {rec:.4f}")
    print(f"f1-Score: {f1:.4f}")
    print(f"Auc-Roc: {auc:.4f}")
    print('\n')
    print(classification_report(y_test, y_pred, target_names=["Non-défaut", "Défaut"]))

    # matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Non-défaut", "Défaut"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes,
           ylabel="Vraie classe", xlabel="Classe prédite",
           title="Matrice de confusion - Neural Network")
    seuil = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > seuil else "black", fontsize=14)
    plt.tight_layout()
    #plt.savefig("resultats/confusion_neural_network.png", dpi=150, bbox_inches="tight")
    #print("Sauvegardé: resultats/confusion_neural_network.png")
    #plt.close()


    # Partie 5
    print("\n5. Cross-Validation du Deep Learning (5-Fold)")

    # combiner train et val pour la cv
    X_cv = np.vstack([X_train.values, X_val.values])
    y_cv = np.concatenate([y_train.values, y_val.values])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_cv = []

    for fold, (idx_train, idx_val) in enumerate(cv.split(X_cv, y_cv)):
        model_fold = ReseauCredit(nb_features, couches=dims, dropout=bp["dropout"])
        criterion_fold = FocalLoss(alpha=bp["alpha"], gamma=bp["gamma"])


        optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=bp["lr"])
        X_f_tr = torch.FloatTensor(X_cv[idx_train])
        y_f_tr = torch.FloatTensor(y_cv[idx_train])
        X_f_val = torch.FloatTensor(X_cv[idx_val])
        y_f_val = torch.FloatTensor(y_cv[idx_val])

        dataset_f = TensorDataset(X_f_tr, y_f_tr)
        loader_f = DataLoader(dataset_f, batch_size=bp["batch_size"], shuffle=True)

        model_fold.train()
        for epoch in range(60):
            for bx, by in loader_f:
                optimizer_fold.zero_grad()
                loss = criterion_fold(model_fold(bx).squeeze(), by)
                loss.backward()
                optimizer_fold.step()

        # Évaluer
        model_fold.eval()
        with torch.no_grad():
            proba_fold = torch.sigmoid(model_fold(X_f_val).squeeze()).numpy()
        auc_fold = roc_auc_score(y_cv[idx_val], proba_fold)
        scores_cv.append(auc_fold)
        print(f"Fold {fold+1}: AUC-ROC = {auc_fold:.4f}")

    print(f"Moyenne Cv:{np.mean(scores_cv):.4f} ( {np.std(scores_cv):.4f})")

    # les résultats
    metriques = {
        "Neural Network (Focal Loss)": {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1_score": float(f1),
            "auc_roc": float(auc),
            "cv_auc_mean": float(np.mean(scores_cv)),
            "cv_auc_std": float(np.std(scores_cv)),
        }
    }

    with open("resultats/metriques_deep_learning.json", "w") as f:
        json.dump(metriques, f, indent=2, ensure_ascii=False)

    return metriques, y_proba

if __name__ == "__main__":
    metriques, _ = entrainer_deep_learning()