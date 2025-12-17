"""
Script d'entraînement du modèle XGBoost pour la détection de fraude bancaire
Script de production pour générer model.pkl et scaler.pkl
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib


def load_data(data_path):
    """
    Charge les données depuis le fichier CSV

    Args:
        data_path (str): Chemin vers le fichier creditcard.csv

    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    print(f"Chargement des données depuis {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def preprocess_data(df):
    """
    Prétraitement des données:
    - Suppression de la colonne Time
    - Standardisation de la colonne Amount

    Args:
        df (pd.DataFrame): DataFrame brut

    Returns:
        tuple: (X, y, scaler)
    """
    print("\nPrétraitement des données...")

    # Copie pour éviter les modifications sur l'original
    df_processed = df.copy()

    # Suppression de la colonne Time
    if 'Time' in df_processed.columns:
        df_processed = df_processed.drop('Time', axis=1)
        print("- Colonne 'Time' supprimée")

    # Séparation features et target
    X = df_processed.drop('Class', axis=1)
    y = df_processed['Class']

    # Standardisation de Amount
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    print("- Colonne 'Amount' standardisée")

    print(f"Distribution des classes - Fraude: {y.sum()} ({y.sum()/len(y)*100:.2f}%), Normal: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.2f}%)")

    return X, y, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split train/test avec stratification

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion du test set
        random_state (int): Seed pour reproductibilité

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\nSplit des données (train: {int((1-test_size)*100)}%, test: {int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"- Train set: {X_train.shape[0]} lignes")
    print(f"- Test set: {X_test.shape[0]} lignes")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state=42):
    """
    Application de SMOTE pour équilibrer les classes sur le train set

    Args:
        X_train (pd.DataFrame): Features d'entraînement
        y_train (pd.Series): Target d'entraînement
        random_state (int): Seed pour reproductibilité

    Returns:
        tuple: (X_train_resampled, y_train_resampled)
    """
    print("\nApplication de SMOTE sur le train set...")
    print(f"Avant SMOTE - Fraude: {y_train.sum()}, Normal: {len(y_train)-y_train.sum()}")

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Après SMOTE - Fraude: {y_train_resampled.sum()}, Normal: {len(y_train_resampled)-y_train_resampled.sum()}")

    return X_train_resampled, y_train_resampled


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Entraînement du modèle XGBoost

    Args:
        X_train (pd.DataFrame): Features d'entraînement
        y_train (pd.Series): Target d'entraînement
        X_test (pd.DataFrame): Features de test
        y_test (pd.Series): Target de test

    Returns:
        xgb.XGBClassifier: Modèle entraîné
    """
    print("\nEntraînement du modèle XGBoost...")

    # Paramètres du modèle
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    # Entraînement
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print("Modèle entraîné avec succès!")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Évaluation du modèle sur le test set

    Args:
        model: Modèle entraîné
        X_test (pd.DataFrame): Features de test
        y_test (pd.Series): Target de test
    """
    print("\n" + "="*50)
    print("ÉVALUATION DU MODÈLE")
    print("="*50)

    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Métriques
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))

    print("\nMatrice de confusion:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}")


def save_model(model, scaler, models_dir='../models'):
    """
    Sauvegarde du modèle et du scaler pour la Phase 2

    Args:
        model: Modèle entraîné
        scaler: Scaler utilisé pour Amount
        models_dir (str): Répertoire de sauvegarde
    """
    # Création du dossier models s'il n'existe pas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, models_dir)
    os.makedirs(models_path, exist_ok=True)

    # Chemins de sauvegarde (noms standardisés pour Phase 2)
    model_path = os.path.join(models_path, 'model.pkl')
    scaler_path = os.path.join(models_path, 'scaler.pkl')

    # Sauvegarde
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nModèles sauvegardés pour la Phase 2:")
    print(f"  ✓ Modèle: {model_path}")
    print(f"  ✓ Scaler: {scaler_path}")


def main():
    """
    Fonction principale d'entraînement
    """
    # Chemin vers les données (depuis src/ vers data/raw/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'raw', 'creditcard.csv')

    # Vérification de l'existence du fichier
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Le fichier {data_path} n'existe pas.\n"
            "Veuillez placer le fichier creditcard.csv dans le dossier data/raw/"
        )

    # Pipeline d'entraînement
    df = load_data(data_path)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    model = train_xgboost(X_train_resampled, y_train_resampled, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    save_model(model, scaler)

    print("\n" + "="*50)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*50)


if __name__ == "__main__":
    main()
