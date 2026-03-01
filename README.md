# Prédiction de Défaut de Paiement - Carte de Crédit
Anesti Gospodinov M2 MBFA ARM
## Description
Projet  ML : prédire si un client va faire défaut sur sa carte de crédit.
On compare des modèles classiques (Logistic Regression, Random Forest, XGBoost, LightGBM)
avec un réseau de neurones PyTorch qui utilise une Focal Loss personnalisée.

**Dataset** : Default of Credit Card Clients (UCI)
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

## Structure du projet

```
projet_final/
data/credit_card_default.csv  le dataset (à télécharger)
01_eda.ipynb                  exploration des données
pretraitement.py              nettoyage + features + split
entrainement_classique.py     modèles sklearn/xgboost/lightgbm + optuna
entrainement_deep.py          réseau de neurones PyTorch + Focal Loss custom
main.py                       lance tout et compare les résultats
resultats/                    graphiques et métriques générés
rapport.pdf                   rapport final
requirements.txt
README.md
```

## Installation et lancement

```bash
pip install -r requirements.txt
python main.py
```