import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1 : Charger les données
def charger_donnees(chemin="data/credit_card_default.csv"):

    df = pd.read_csv(chemin)

    # nettoyer les noms de colonnes
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # renommer la colonne cible
    for ancien_nom in ["default.payment.next.month", "default_payment_next_month", "y"]:
        if ancien_nom in df.columns:
            df = df.rename(columns={ancien_nom: "default"})
            break

    # supprimer la colonne ID
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Non-défaut: {(df['default'] == 0).sum()} ({(df['default'] == 0).mean():.1%})")
    print(f"défaut: {(df['default'] == 1).sum()} ({(df['default'] == 1).mean():.1%})")
    print(f"Valeurs manquantes : {df.isnull().sum().sum()}")

    return df

# 2 : Nettoyer les données
def nettoyer_donnees(df):

    print("\n1. Nettoyage des données")

    df = df.copy()

    # Corriger education
    nb_edu_bizarre = df["education"].isin([0, 5, 6]).sum()
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    print(f"Education: {nb_edu_bizarre} valeurs corrigées (0,5 et 6 - autre)")

    # Corriger marriage
    nb_mar_bizarre = (df["marriage"] == 0).sum()
    df["marriage"] = df["marriage"].replace({0: 3})
    print(f"Marriage : {nb_mar_bizarre} 0 - autre")

    # Corriger les statuts de paiement
    colonnes_pay = [c for c in df.columns if c.startswith("pay_") and "amt" not in c]
    nb_pay_corrige = 0
    for col in colonnes_pay:
        mask = df[col] < -1
        nb_pay_corrige += mask.sum()
        df.loc[mask, col] = -1
    print(f"Statuts de paiement : {nb_pay_corrige}")

    # Vérifier les doublons
    nb_doublons = df.duplicated().sum()
    if nb_doublons > 0:
        df = df.drop_duplicates()
        print(f"{nb_doublons} doublons supprimés")
    else:
        print(f"Pas de doublons")
    return df

#3 : Créer des nouvelles variables
def creer_features(df):

    print("\n2. Feature engineering")
    df = df.copy()

    # Taux d'utilisation du crédit
    colonnes_facture = [f"bill_amt{i}" for i in range(1, 7)]
    facture_moyenne = df[colonnes_facture].mean(axis=1)
    limite = df["limit_bal"].replace(0, np.nan)  # éviter division par 0
    df["taux_utilisation"] = (facture_moyenne / limite).fillna(0)
    print(f"taux_utilisation : moyenne = {df['taux_utilisation'].mean():.3f}")

    # Nombre de mois en retard
    colonnes_pay = [c for c in df.columns if c.startswith("pay_") and "amt" not in c]
    df["nb_mois_retard"] = (df[colonnes_pay] > 0).sum(axis=1)
    print(f"nb_mois_retard : moyenne = {df['nb_mois_retard'].mean():.2f}")

    # Retard maximum
    df["retard_max"] = df[colonnes_pay].max(axis=1)
    print(f"retard_max : max observé = {df['retard_max'].max()}")

    # Jamais en retard (oui/non)
    df["jamais_en_retard"] = (df["nb_mois_retard"] == 0).astype(int)
    print(f"jamais_en_retard : {df['jamais_en_retard'].mean():.1%} des clients")

    # Ratio paiement
    colonnes_paiement = [f"pay_amt{i}" for i in range(1, 7)]
    total_paye = df[colonnes_paiement].sum(axis=1)
    total_facture = df[colonnes_facture].sum(axis=1).replace(0, np.nan)
    df["ratio_paiement"] = (total_paye / total_facture).fillna(0).clip(0, 5)
    print(f"ratio_paiement : moyenne = {df['ratio_paiement'].mean():.3f}")

    # Tendance des factures (la pente)
    factures = df[colonnes_facture].values
    x = np.arange(6)
    x_centre = x - x.mean()
    factures_centre = factures - factures.mean(axis=1, keepdims=True)
    df["tendance_factures"] = (factures_centre * x_centre).sum(axis=1) / (x_centre ** 2).sum()
    print(f"tendance_factures : créée")

    # Score de risque composite
    df["score_risque"] = (
        df["nb_mois_retard"] * 2
        + df["taux_utilisation"].clip(0, 3) * 1.5
        - df["ratio_paiement"].clip(0, 2) * 1
    )
    print(f"score_risque : moyenne = {df['score_risque'].mean():.2f}")
    print(f"total : {df.shape[1]} colonnes")
    return df

# 4 : Encoder les variables catégorielles
def encoder_categories(df):

    print("\n3. Encodage des catégories")

    df = df.copy()
    colonnes_cat = ["sex", "education", "marriage"]
    colonnes_existantes = [c for c in colonnes_cat if c in df.columns]

    avant = df.shape[1]
    df = pd.get_dummies(df, columns=colonnes_existantes, drop_first=True, dtype=int)
    apres = df.shape[1]

    print(f"{avant} colonnes avant -> {apres} colonnes après")
    print(f"Colonnes créées : {[c for c in df.columns if any(c.startswith(x) for x in colonnes_cat)]}")

    return df

# 5 : Séparer et normaliser les données
def separer_et_normaliser(df):

    print("\n4. Séparation et normalisation")

    X = df.drop(columns=["default"])
    y = df["default"]

    # 1er split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # 2ème split
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp

    )

    print(f"Train : {X_train.shape[0]} samples (défaut: {y_train.mean():.1%})")
    print(f"Validation : {X_val.shape[0]} samples (défaut: {y_val.mean():.1%})")
    print(f"Test: {X_test.shape[0]} samples (défaut: {y_test.mean():.1%})")

    # Normalisation
    colonnes_num = [
        "limit_bal", "age",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
        "taux_utilisation", "ratio_paiement", "tendance_factures", "score_risque",
    ]
    colonnes_num = [c for c in colonnes_num if c in X_train.columns]

    scaler = StandardScaler()
    X_train[colonnes_num] = scaler.fit_transform(X_train[colonnes_num])  # FIT
    X_val[colonnes_num] = scaler.transform(X_val[colonnes_num])
    X_test[colonnes_num] = scaler.transform(X_test[colonnes_num])

    print(f"{len(colonnes_num)} colonnes normalisées")
    print("Scaler fit sur Train uniquement (pas de data leakage)")
    print("Séparation et normalisation terminées")

    return X_train, X_val, X_test, y_train, y_val, y_test

# lance tout le prétraitement
def lancer_pretraitement(chemin="data/credit_card_default.csv"):

    df = charger_donnees(chemin)
    df = nettoyer_donnees(df)
    df = creer_features(df)
    df = encoder_categories(df)
    X_train, X_val, X_test, y_train, y_val, y_test = separer_et_normaliser(df)

    print(f"\n5. Résumé: {X_train.shape[1]} features, {X_train.shape[0]} samples train")

    return X_train, X_val, X_test, y_train, y_val, y_test

# Si on lance ce fichier directement
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = lancer_pretraitement()
    print(f"X_train shape: {X_train.shape}")
    print(f"Colonnes: {X_train.columns.tolist()}")