import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
file_path = "Campus_Selection.csv"
campus_df = pd.read_csv(file_path)

# Data understanding
print("\nColumn names of the dataset:")
print(campus_df.columns)

print("First few rows of the dataset:")
print(campus_df.head())

print("\nBasic information about the dataset:")
print(campus_df.info())

print("\nSummary statistics of the dataset:")
print(campus_df.describe())

print("\nMissing values in the dataset:")
print(campus_df.isnull().sum())

# Distribution of target classes
print("\nDistribution of 'status' classes:")
print(campus_df['status'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(x='status', data=campus_df)
plt.title('Distribution of Placement Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Numeric columns pairplot
sns.pairplot(campus_df, hue='status')
plt.suptitle('Pairplot of Campus Selection Dataset', y=1.02)
plt.show()

# Compute the correlation matrix
numeric_columns = campus_df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = campus_df[numeric_columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Boxplots of numeric features by 'status'
plt.figure(figsize=(12, 8))
numeric_columns = campus_df.select_dtypes(include=['float64']).columns
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='status', y=column, data=campus_df)
plt.suptitle('Boxplots of Numeric Features by Status')
plt.tight_layout()
plt.show()

# Histograms of numeric features
plt.figure(figsize=(12, 8))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=campus_df, x=column, kde=True)
plt.suptitle('Histograms of Numeric Features')
plt.tight_layout()
plt.show()

# Perform label encoding for categorical variables
label_encoders = {}
categorical_columns = campus_df.select_dtypes(include=['object']).columns.drop('status')
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    campus_df[column] = label_encoders[column].fit_transform(campus_df[column])


# Splitting the dataset into features (X) and target variable (y)
X = campus_df.drop(columns=['status'])
y = campus_df['status']

# Splitting the dataset into the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Store X_train in a CSV file
X_train.to_csv("X_train.csv", index=False)

# Store X_train_scaled in a CSV file
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("X_train_scaled.csv", index=False)

# Store X_train_pca in a CSV file
pd.DataFrame(X_train_pca, columns=[f"PCA_{i+1}" for i in range(X_train_pca.shape[1])]).to_csv("X_train_pca.csv", index=False)

# Similarly, you can store other variables like X_test, X_test_scaled, X_test_pca, y_train, y_test, etc.


# Model Training and Evaluation
models = {
    "Random Forest Classifier": RandomForestClassifier(),
    "Naive Bayes Classifier": GaussianNB(),
    "SVM Classifier": SVC()
}

trained_models = {}
evaluation_metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1-score": f1_score
}

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    trained_models[name] = model
    print(f"Training of {name} completed.")

print("All models trained successfully.")

for name, model in trained_models.items():
    print(f"\n{name}:")
    y_pred = model.predict(X_test_pca)

    metrics_values = {}
    for metric_name, metric_func in evaluation_metrics.items():
        if metric_name == "Accuracy":
            metric_value = metric_func(y_test, y_pred)
        else:
            metric_value = metric_func(y_test, y_pred, average='weighted')
        metrics_values[metric_name] = metric_value

    for metric_name, metric_value in metrics_values.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Hyperparameter Tuning with PCA
param_grid = {
    "Random Forest Classifier": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
    "SVM Classifier": {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    "Naive Bayes Classifier": {}
}

evaluation_results = []

best_models = {}

for name, model in models.items():
    print(f"Hyperparameter tuning for {name}...")
    clf = GridSearchCV(model, param_grid[name], scoring='accuracy', cv=5)
    clf.fit(X_train_pca, y_train)
    best_models[name] = clf.best_estimator_
    print(f"Best parameters: {clf.best_params_}, Best score: {clf.best_score_:.4f}")

    # Perform predictions on the test set
    y_pred = best_models[name].predict(X_test_pca)

    # Calculate evaluation metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Append evaluation results to the list
    evaluation_results.append({
        "Model": name,
        "F1 Score": f1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Confusion Matrix": confusion_mat
    })

    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print("Confusion Matrix:")
    print(confusion_mat)
    print()

# Convert evaluation results to a DataFrame
evaluation_df = pd.DataFrame(evaluation_results)

# Save the evaluation results to Excel
evaluation_df.to_csv("evaluation_results_hyperparameter_tuning.csv", index=False)

print("Evaluation results after hyperparameter tuning saved to 'evaluation_results_hyperparameter_tuning.csv' file.")

# Print explained variance ratio of PCA components
print("Explained Variance Ratio of PCA Components:")
print(pca.explained_variance_ratio_)
