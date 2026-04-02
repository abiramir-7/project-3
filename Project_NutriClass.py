# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the dataset 
df = pd.read_csv('env/synthetic_food_dataset_imbalanced.csv')

print("--- Data Overview ---")
print(f"Original shape: {df.shape}")

# 2. DATA CLEANING (The Fix for your Error)
# Checking for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Dropping rows with any missing values
df = df.dropna()
print(f"Shape after dropping NaNs: {df.shape}")

# 3. Exploratory Data Analysis (EDA) 
print("\nGenerating Distribution Plot...")
plt.figure(figsize=(10,6))
sns.countplot(y='Food_Name', data=df)
plt.title('Distribution of Food Categories')
plt.show()

# 4. Preprocessing 
le = LabelEncoder()
categorical_cols = ['Meal_Type', 'Preparation_Method', 'Is_Vegan', 'Is_Gluten_Free']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(['Food_Name'], axis=1)
y = df['Food_Name']

# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (StandardScaler needs clean data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 5. Model Comparison
print("\n--- Testing Multiple Models ---")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, m in models.items():
    m.fit(X_train, y_train)
    score = m.score(X_test, y_test)
    print(f"{name} Accuracy: {score:.2f}")

# 6. Final Model & Feature Importance
# We use Random Forest for the final analysis as it usually performs best
final_model = models["Random Forest"]
importances = final_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.title('Nutritional Attributes Importance (Random Forest)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# 7. Final Evaluation (Confusion Matrix)
y_pred = final_model.predict(X_test)

print("\n--- Final Model Results (Random Forest) ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=final_model.classes_, yticklabels=final_model.classes_, cmap='Blues')
plt.title('Confusion Matrix: Predicted vs Actual Foods')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()