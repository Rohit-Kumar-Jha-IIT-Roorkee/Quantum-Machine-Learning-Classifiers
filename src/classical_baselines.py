
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, recall_score
import sys

def train_baselines(data_path='processed_data.npz'):
    print(f"Loading data from {data_path}...")
    try:
        data = np.load(data_path)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Run data_loader.py first.")
        sys.exit(1)
        
    print(f"Data Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        # MLP with similar parameter count to a simple VQC (approx 4 inputs -> faint hidden -> 1 output)
        # Using a small architecture to be "fair" per problem statement
        "MLP (Small)": MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*40)
    print("CLASSICAL BASELINES (4 Features)")
    print("="*40)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        rec = recall_score(y_test, y_pred, pos_label=1)
        
        results[name] = {"Accuracy": acc, "AUC": auc, "Recall": rec}
        
        print(f"--> Accuracy: {acc:.4f}")
        print(f"--> AUC-ROC:  {auc:.4f}")
        print(f"--> Recall:   {rec:.4f}")
        print("--> Report:")
        print(classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"]))
        
    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    for name, metrics in results.items():
        print(f"{name:<20} | Acc: {metrics['Accuracy']:.4f} | AUC: {metrics['AUC']:.4f} | Rec: {metrics['Recall']:.4f}")

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'processed_data.npz'
    train_baselines(data_file)
