import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

print(data.head())

print(data.info())

X = data.drop("target", axis=1)
y = data["target"]

# train set * 0,3 = test sete
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

numerical_cols = X.columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)
    ])

# Decision Tree
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=61))
])
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=61))
])
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Calculation of performance metrics
def evaluate_model(y_test, y_pred):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, average='weighted')
    }

dt_metrics = evaluate_model(y_test, y_pred_dt)

rf_metrics = evaluate_model(y_test, y_pred_rf)

conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

disp_dt = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_dt, display_labels=iris.target_names)
disp_dt.plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()

disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=iris.target_names)
disp_rf.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Visualizing the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dt_model.named_steps['classifier'], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Result
results = pd.DataFrame([dt_metrics, rf_metrics], index=["Decision Tree", "Random Forest"])
print(results)
