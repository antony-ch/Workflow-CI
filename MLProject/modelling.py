# Membangun_model/modelling_tuning.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import mlflow
import mlflow.sklearn
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub # Pastikan dagshub diimport
dagshub.init(repo_owner='antony-ch',
             repo_name='Eksperimen_SML_AntonyCH',
             mlflow=True)

# BLOK BERMASALAH SEBELUMNYA (import mlflow dan with mlflow.start_run(): ...) TELAH DIHAPUS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfigurasi MLflow untuk DagsHub ---
# Baris ini masih dipertahankan untuk memastikan MLflow_TRACKING_URI diatur
# meskipun dagshub.init() juga melakukan hal serupa. Ini tidak akan menyebabkan konflik.
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/antony-ch/Eksperimen_SML_AntonyCH.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'antony-ch' # Ganti dengan username DagsHub Anda
# UNTUK KEAMANAN: JANGAN hardcode password/PAT di sini.
# Atur sebagai variabel lingkungan sebelum menjalankan script, misal:
# export MLFLOW_TRACKING_PASSWORD='<your_dagshub_pat>' (Linux/macOS)
# $env:MLFLOW_TRACKING_PASSWORD='<your_dagshub_pat>' (PowerShell Windows)
# Atau, yang paling disarankan untuk development lokal: `dagshub login` di terminal sebelum menjalankan script.

logging.info("MLflow tracking URI diatur ke DagsHub.")

def load_data(features_path, target_path):
    """Memuat dataset yang sudah diproses."""
    logging.info(f"Memuat fitur dari: {features_path}")
    X = pd.read_csv(features_path)
    logging.info(f"Memuat target dari: {target_path}")
    y = pd.read_csv(target_path).squeeze() # .squeeze() untuk memastikan jadi Series
    return X, y

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, params):
    """Melatih dan mengevaluasi model, logging ke MLflow."""
    with mlflow.start_run(run_name=model_name):
        logging.info(f"Memulai run MLflow untuk model: {model_name}")

        # Log parameters
        mlflow.log_params(params)
        logging.info(f"Parameter logged: {params}")

        # Train model
        model.fit(X_train, y_train)
        logging.info(f"Model {model_name} berhasil dilatih.")

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] # Probability for ROC AUC

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log basic metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)
        logging.info(f"Metrics logged: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc:.4f}")

        # --- Manual Logging: Tambahan 2 Metrik (Minimal) ---
        # Untuk confusion_matrix: (TN, FP, FN, TP) = array.ravel()
        # TN = True Negatives, FP = False Positives, FN = False Negatives, TP = True Positives
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # 1. Negative Predictive Value (NPV)
        # NPV = TN / (TN + FN)
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0
        mlflow.log_metric("negative_predictive_value", npv)
        logging.info(f"Tambahan Metrik: Negative Predictive Value (NPV)={npv:.4f}")

        # 2. Specificity (True Negative Rate)
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        mlflow.log_metric("specificity", specificity)
        logging.info(f"Tambahan Metrik: Specificity={specificity:.4f}")

        # Log confusion matrix as an artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close() # Penting untuk menutup figure setelah menyimpan
        logging.info("Confusion matrix logged as artifact.")

        # Log ROC Curve as an artifact
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--') # Garis diagonal
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close() # Penting untuk menutup figure setelah menyimpan
        logging.info("ROC curve logged as artifact.")

        # Log the model
        mlflow.sklearn.log_model(model, "model")
        logging.info(f"Model {model_name} logged to MLflow.")
        logging.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    logging.info(f"Run MLflow untuk {model_name} selesai.")

if __name__ == "__main__":
    logging.info("Memulai script modelling_tuning.py...")

    # Path ke dataset yang sudah dipreproses (relatif terhadap folder Membangun_model)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, 'namadataset_preprocessing')
    features_file = os.path.join(data_folder, 'telco_customer_churn_features_preprocessed.csv')
    target_file = os.path.join(data_folder, 'telco_customer_churn_target_preprocessed.csv')

    # Load data
    X, y = load_data(features_file, target_file)
    logging.info(f"Dimensi data loaded: X={X.shape}, y={y.shape}")

    # Split data
    # Menggunakan stratify=y penting untuk menjaga proporsi kelas target (churn/no churn)
    # di training dan testing set, terutama jika ada class imbalance.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info("Data berhasil dibagi menjadi training dan testing set.")

    # --- Hyperparameter Tuning dengan GridSearchCV ---
    # GridSearchCV akan mencari kombinasi parameter terbaik dari grid yang diberikan.
    # cv=3 berarti 3-fold cross-validation.
    # scoring='f1' dipilih karena masalah class imbalance; F1-score adalah rata-rata harmonik Precision dan Recall.
    # n_jobs=-1 berarti menggunakan semua core CPU yang tersedia.
    # verbose=1 akan menampilkan progres.

    # Logistic Regression
    lr_param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0], # Regularization strength
        'solver': ['liblinear'], # 'liblinear' mendukung L1 dan L2 penalty
        'penalty': ['l1', 'l2'] # Tipe regularisasi
    }
    lr_grid_search = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), # Meningkatkan max_iter
                                  lr_param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    logging.info("Memulai GridSearchCV untuk Logistic Regression...")
    lr_grid_search.fit(X_train, y_train)
    best_lr_model = lr_grid_search.best_estimator_
    best_lr_params = lr_grid_search.best_params_
    logging.info(f"Best params for Logistic Regression: {best_lr_params}")
    train_and_evaluate(best_lr_model, X_train, y_train, X_test, y_test, "Optimized Logistic Regression", best_lr_params)

    # Decision Tree
    dt_param_grid = {
        'max_depth': [None, 5, 10, 15], # Kedalaman maksimum pohon
        'min_samples_split': [2, 5, 10], # Jumlah minimum sampel yang dibutuhkan untuk membagi sebuah node
        'min_samples_leaf': [1, 2, 4] # Jumlah minimum sampel yang dibutuhkan di node daun
    }
    dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                                  dt_param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    logging.info("Memulai GridSearchCV untuk Decision Tree...")
    dt_grid_search.fit(X_train, y_train)
    best_dt_model = dt_grid_search.best_estimator_
    best_dt_params = dt_grid_search.best_params_
    logging.info(f"Best params for Decision Tree: {best_dt_params}")
    train_and_evaluate(best_dt_model, X_train, y_train, X_test, y_test, "Optimized Decision Tree", best_dt_params)

    # Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200, 300], # Jumlah pohon dalam forest
        'max_depth': [10, 20, None], # Kedalaman maksimum setiap pohon
        'min_samples_split': [5, 10] # Sama seperti Decision Tree
    }
    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                                  rf_param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    logging.info("Memulai GridSearchCV untuk Random Forest...")
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    best_rf_params = rf_grid_search.best_params_
    logging.info(f"Best params for Random Forest: {best_rf_params}")
    train_and_evaluate(best_rf_model, X_train, y_train, X_test, y_test, "Optimized Random Forest", best_rf_params)

    logging.info("Proses pelatihan model dan logging MLflow selesai.")