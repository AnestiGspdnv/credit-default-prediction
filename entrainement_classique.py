import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

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

def entrainer_modeles_classiques():
    print("#Modèles Classiques")

    # Charger les données prétraitées
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

    print("\n2 : Optimisation des hyperparamètres -Optuna")

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