import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import optuna

# pour moins de bruit
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
from pretraitement import lancer_pretraitement

# Évaluer de modèle
def evaluer_modele(nom, modele, X_test, y_test):

    y_pred = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)

    if y_proba.ndim == 2: #on prend la colonne défaut
        y_proba = y_proba[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"{nom}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Non-défaut", "Défaut"]))

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_score": f1, "auc_roc": auc,
        "y_pred": y_pred, "y_proba": y_proba
    }

# cv
def cross_validation_rapide(nom, modele, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(modele, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    print(f"{nom:25s}  AUC-ROC CV: {scores.mean():.4f} (± {scores.std():.4f})")
    return scores.mean()

# Tracer la matrice de confusion
def tracer_matrice_confusion(nom, y_test, y_pred, chemin=None):

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Non-défaut", "Défaut"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes,
           ylabel="Vraie classe", xlabel="Classe prédite",
           title=f"Matrice de confusion - {nom}")

    seuil = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > seuil else "black", fontsize=14)

    plt.tight_layout()
    if chemin:
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        plt.savefig(chemin, dpi=150, bbox_inches="tight")
        print(f"Sauvegardé: {chemin}")
    plt.close()

# principal
def entrainer_modeles_classiques(X_train=None, X_val=None, X_test=None,y_train=None, y_val=None, y_test=None):

    # Charger les données prétraitées
    if X_train is None:
        X_train, X_val, X_test, y_train, y_val, y_test = lancer_pretraitement()

    # Partie 1 :
    print("\n1 : Modèles Baseline par défaut")
    modeles_baseline = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, random_state=42, scale_pos_weight=3.5,
            eval_metric="logloss", use_label_encoder=False, verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, random_state=42, is_unbalance=True, verbose=-1
        ),
    }

    print("Cross-Validation donc 5 folds stratifiés :")
    for nom, modele in modeles_baseline.items():
        cross_validation_rapide(nom, modele, X_train, y_train)

    # Partie 2
    print("\n2 : Optimisation des hyperparamètres (Optuna)")

    # Logistic Regression

    def objectif_lr(trial):
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "saga" if penalty == "l1" else "lbfgs"
        model = LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            max_iter=2000, random_state=42, class_weight="balanced"
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study_lr = optuna.create_study(direction="maximize")
    study_lr.optimize(objectif_lr, n_trials=25)
    print(f"Meilleur AUC: {study_lr.best_value:.4f} Params : {study_lr.best_params}")

    # Random Forest
    print("\nOptimisation Random Forest...")

    def objectif_rf(trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objectif_rf, n_trials=25)
    print(f"  Meilleur AUC : {study_rf.best_value:.4f} Params : {study_rf.best_params}")

    # XGBoost
    print("\nOptimisation XGBoost...")

    def objectif_xgb(trial):
        model = xgb.XGBClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            eval_metric="logloss", random_state=42, use_label_encoder=False, verbosity=0
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(objectif_xgb, n_trials=30)
    print(f"Meilleur AUC: {study_xgb.best_value:.4f} Params : {study_xgb.best_params}")

    # LightGBM
    print("\nOptimisation LightGBM...")

    def objectif_lgbm(trial):
        model = lgb.LGBMClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 150),
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            is_unbalance=True, random_state=42, verbose=-1
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(objectif_lgbm, n_trials=30)
    print(f"Meilleur AUC: {study_lgbm.best_value:.4f} Params : {study_lgbm.best_params}")


    # Partie 3
    print("\n3 : Entraînement final et Evaluation")

    # Reconstruire les modèles avec les meilleurs paramètres
    params_lr = study_lr.best_params
    solver_lr = "saga" if params_lr.get("penalty") == "l1" else "lbfgs"
    modeles_optimises = {
        "Logistic Regression": LogisticRegression(
            **params_lr, solver=solver_lr, max_iter=2000,
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            **study_rf.best_params, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            **study_xgb.best_params, eval_metric="logloss",
            random_state=42, use_label_encoder=False, verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            **study_lgbm.best_params, is_unbalance=True,
            random_state=42, verbose=-1
        ),
    }

    # entraîner et évaluer chaque modèle
    tous_les_resultats = {}
    donnees_roc = {}

    for nom, modele in modeles_optimises.items():
        modele.fit(X_train, y_train)
        resultats = evaluer_modele(nom, modele, X_test, y_test)
        tous_les_resultats[nom] = {
            k: float(v) for k, v in resultats.items()
            if k not in ["y_pred", "y_proba"]
        }
        donnees_roc[nom] = {"y_true": y_test, "y_proba": resultats["y_proba"]}

        # Matrice de confusion
        nom_fichier = nom.lower().replace(" ", "_")
        tracer_matrice_confusion(
            nom, y_test, resultats["y_pred"],
            chemin=f"resultats/confusion_{nom_fichier}.png"
        )


    # Partie 4 : Courbes ROC comparatives

    print("\n4. Courbes ROC")

    fig, ax = plt.subplots(figsize=(8, 6))
    for nom, data in donnees_roc.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        auc_val = roc_auc_score(data["y_true"], data["y_proba"])
        ax.plot(fpr, tpr, label=f"{nom} (AUC={auc_val:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Aléatoire (AUC=0.500)")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC - Modèles Classiques")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("resultats", exist_ok=True)
    plt.savefig("resultats/courbes_roc_classiques.png", dpi=150, bbox_inches="tight")
    plt.close()


    # Partie 5 :
    print(" Tableau récapitulatif")
    recap = pd.DataFrame(tous_les_resultats).T.round(4)
    print(recap.to_string())

    meilleur = max(tous_les_resultats, key=lambda k: tous_les_resultats[k]["auc_roc"])
    print(f"\nMeilleur modèle classique : {meilleur}")
    print(f"AUC-ROC = {tous_les_resultats[meilleur]['auc_roc']:.4f}")

    # Sauvegarder les métriques en json
    with open("resultats/metriques_classiques.json", "w") as f:
        json.dump(tous_les_resultats, f, indent=2, ensure_ascii=False)

    return tous_les_resultats, donnees_roc

if __name__ == "__main__":
    resultats, roc_data = entrainer_modeles_classiques()
