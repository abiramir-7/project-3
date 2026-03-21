# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset 
df = pd.read_csv('env/synthetic_food_dataset_imbalanced.csv')

print("--- Data Overview ---")
print(df.head())
print(df.info())

# Exploratory Data Analysis (EDA) 
print("\nGenerating Visualizations...")
plt.figure(figsize=(10,6))
sns.countplot(y='Food_Name', data=df)
plt.title('Distribution of Food Categories')
plt.show()

# Preprocessing 
le = LabelEncoder()
df['Meal_Type'] = le.fit_transform(df['Meal_Type'])
df['Preparation_Method'] = le.fit_transform(df['Preparation_Method'])
df['Is_Vegan'] = le.fit_transform(df['Is_Vegan'])
df['Is_Gluten_Free'] = le.fit_transform(df['Is_Gluten_Free'])

X = df.drop(['Food_Name'], axis=1)
y = df['Food_Name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building and Evaluation
print("\nTraining Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Final Model Results ---")
print(classification_report(y_test, y_pred))
