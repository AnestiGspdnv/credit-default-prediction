import json
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def main():
    print("\n" + "#" * 65)
    print("#                                                               #")
    print("#                Anesti Gospodinov M2 MBFA ARM                  #")
    print("#         Projet de Prediction de default de paiement           #")
    print("#            Comparaison Classique vs Deep Learning             #")
    print("#                                                               #")
    print("#" * 65)

    # Partie PRÉTRAITEMENT
    print("\n################  PARTIE 1 : PRÉTRAITEMENT  ################")
    from pretraitement import lancer_pretraitement
    X_train, X_val, X_test, y_train, y_val, y_test = lancer_pretraitement()

    # PARTIE 2 : Modèles classiques
    print("\n################  PARTIE 2 : MODÈLES CLASSIQUES  ################")
    from entrainement_classique import entrainer_modeles_classiques
    resultats_classiques, _ = entrainer_modeles_classiques(X_train, X_val, X_test, y_train, y_val, y_test)

    # PARTIE 3 : Deep Learning
    print("\n################  PARTIE 3 : DEEP LEARNING + FOCAL LOSS  ################")
    from entrainement_deep import entrainer_deep_learning
    resultats_dl, _ = entrainer_deep_learning(X_train, X_val, X_test, y_train, y_val, y_test)

    # PARTIE 4 : Comparaison finale
    print("\n################  COMPARAISON FINALE DE TOUS LES MODÈLES  ################")

    tous = {}
    tous.update(resultats_classiques)
    for nom, metriques in resultats_dl.items():
        # On garde que les métriques principales (pas cv_auc_mean etc.)
        tous[nom] = {k: v for k, v in metriques.items()
                     if k in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]}

    print("\n" + "-" * 65)
    recap = pd.DataFrame(tous).T.round(4)
    print(recap.to_string())
    print("-" * 65)

    meilleur_nom = max(tous, key=lambda k: tous[k].get("auc_roc", 0))
    meilleur_auc = tous[meilleur_nom]["auc_roc"]
    meilleur_f1 = tous[meilleur_nom]["f1_score"]

    print(f"\n   Meilleur modèle : {meilleur_nom}")
    print(f"     AUC-ROC  = {meilleur_auc:.4f}")
    print(f"     F1-Score = {meilleur_f1:.4f}")

    tous_float = {
        nom: {k: float(v) for k, v in m.items()}
        for nom, m in tous.items()
    }
    with open("resultats/comparaison_finale.json", "w") as f:
        json.dump(tous_float, f, indent=2, ensure_ascii=False)

    print(f"\n  Fichiers générés dans le dossier resultats")
    print("################ PIPELINE TERMINÉ ################")

if __name__ == "__main__":
    main()
