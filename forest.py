import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib.colors import ListedColormap

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Feature Engineering: Creating new features (example: ratio and product of features)
X_new = np.hstack((X, (X[:, 2] / X[:, 3]).reshape(-1, 1), (X[:, 0] * X[:, 1]).reshape(-1, 1)))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning using Grid Search for Random Forest, SVM, k-NN, and Logistic Regression
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    },
    'k-NN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
}

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'k-NN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42)
}

best_models = {}

for model_name, param_grid in param_grids.items():
    print(f"Performing Grid Search for {model_name}...")
    grid_search = GridSearchCV(estimator=models[model_name], param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}\n")

# Evaluate each model
for model_name, model in best_models.items():
    print(f"Evaluating {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}\n")

# Cross-validation for all models
for model_name, model in best_models.items():
    cv_scores = cross_val_score(model, X_new, y, cv=5)
    print(f"Cross-Validation Scores for {model_name}: {cv_scores}")
    print(f"Mean CV Accuracy for {model_name}: {cv_scores.mean():.2f}\n")

# Additional evaluation metrics (using Random Forest as an example)
best_rf_clf = best_models['Random Forest']
y_pred_rf = best_rf_clf.predict(X_test)

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(f"Confusion Matrix for Random Forest:\n{conf_matrix_rf}")

class_report_rf = classification_report(y_test, y_pred_rf)
print(f"Classification Report for Random Forest:\n{class_report_rf}")

# ROC-AUC (applicable for binary classification; using a multiclass strategy for demonstration)
if len(set(y)) > 2:
    y_pred_proba_rf = best_rf_clf.predict_proba(X_test)
    roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr')
    print(f"ROC-AUC Score for Random Forest: {roc_auc_rf:.2f}")

# Visualize decision boundaries (using only first two features for simplicity)
X_reduced = X_new[:, :2]  # Use only the first two features for visualization
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

rf_clf_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf_reduced.fit(X_train_reduced, y_train_reduced)

x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = rf_clf_reduced.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(('red', 'green', 'blue')))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k', s=20, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundaries of Random Forest')
plt.show()
