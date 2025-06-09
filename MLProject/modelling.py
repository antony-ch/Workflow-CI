import pandas as pd
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
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Memulai script. MLflow akan menggunakan konfigurasi dari environment variables.")

def load_data(features_path, target_path):
    """Memuat dataset yang sudah diproses."""
    logging.info(f"Memuat fitur dari: {features_path}")
    X = pd.read_csv(features_path)
    logging.info(f"Memuat target dari: {target_path}")
    y = pd.read_csv(target_path).squeeze()
    return X, y

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, params):
    """Melatih dan mengevaluasi model, logging ke MLflow dalam nested run."""
    logging.info(f"Logging untuk model: {model_name}.")
    logging.info(f"Attempting to log parameters for {model_name}: {params}")

    # --- THIS IS THE CRUCIAL CHANGE ---
    # Log all parameters at once. MLflow handles this efficiently.
    # The keys in 'params' dictionary must be unique for this single call.
    # Since each model now has its own nested run, this is safe.
    mlflow.log_params({str(k): str(v) for k, v in params.items()})
    logging.info(f"Parameters successfully logged for {model_name}.")
    # --- END OF CRUCIAL CHANGE ---

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1_score": f1, "roc_auc_score": roc_auc
    }
    mlflow.log_metrics(metrics)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    mlflow.log_metric("negative_predictive_value", npv)
    mlflow.log_metric("specificity", specificity)

    cm_path = "confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, "plots")
    plt.close()

    roc_path = "roc_curve.png"
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.savefig(roc_path)
    mlflow.log_artifact(roc_path, "plots")
    plt.close()

    mlflow.sklearn.log_model(model, "model")
    logging.info(f"Model {model_name} logged ke MLflow.")

if __name__ == "__main__":
    logging.info("Memulai script modelling.py...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, 'namadataset_preprocessing')

    features_file = os.path.join(data_folder, 'telco_customer_churn_features_preprocessed.csv')
    target_file = os.path.join(data_folder, 'telco_customer_churn_target_preprocessed.csv')

    X, y = load_data(features_file, target_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Logistic Regression ---
    with mlflow.start_run(run_name="Logistic_Regression_Optimization", nested=True):
        logging.info("Melatih dan mengevaluasi Optimized Logistic Regression...")
        lr_param_grid = {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear'], 'penalty': ['l1', 'l2']}
        lr_grid_search = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), lr_param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        lr_grid_search.fit(X_train, y_train)
        train_and_evaluate(lr_grid_search.best_estimator_, X_train, y_train, X_test, y_test, "Optimized Logistic Regression", lr_grid_search.best_params_)

    # --- Decision Tree ---
    with mlflow.start_run(run_name="Decision_Tree_Optimization", nested=True):
        logging.info("Melatih dan mengevaluasi Optimized Decision Tree...")
        dt_param_grid = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        dt_grid_search.fit(X_train, y_train)
        train_and_evaluate(dt_grid_search.best_estimator_, X_train, y_train, X_test, y_test, "Optimized Decision Tree", dt_grid_search.best_params_)

    # --- Random Forest ---
    with mlflow.start_run(run_name="Random_Forest_Optimization", nested=True):
        logging.info("Melatih dan mengevaluasi Optimized Random Forest...")
        rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [5, 10]}
        rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        rf_grid_search.fit(X_train, y_train)
        train_and_evaluate(rf_grid_search.best_estimator_, X_train, y_train, X_test, y_test, "Optimized Random Forest", rf_grid_search.best_params_)

    logging.info("Proses selesai.")