# Food Classification Using Nutritional Data

## Project Overview
NutriClass is a machine learning project designed to classify food items into various categories (such as Pizza, Sushi, Salad, etc.) based on their nutritional attributes. By analyzing features like calories, proteins, carbohydrates, and fats, the model learns the distinct nutritional "fingerprint" of different food types.

## Technical Stack
* **Language:** Python 3.13
* **Editor:** VS Code

## Project Structure
* `Project_NutriClass.py`: The main Python script containing data loading, EDA, preprocessing, and model training.
* `synthetic_food_dataset_imbalanced.csv`: The dataset containing 1,000+ entries of food nutritional data.
* `README.md`: Project documentation.

## How to Run
1. Clone this repository to your local machine.
2. Ensure you have Python installed.
3. Install the required libraries:
4. Run the script:
   *Note: A graph window will appear during execution. Close the window to allow the machine learning model to finish training.*

## Documented Findings

### 1. Data Distribution (EDA)
Upon visualizing the dataset, it was observed that the food categories are **imbalanced**. Some food items (like Pizza) have a significantly higher number of entries compared to others. This reflects real-world scenarios where certain data types are more common than others.

### 2. Feature Importance
Based on the nutritional analysis, the following features played the most critical role in classification:
* **Carbohydrates & Sugar:** Primary indicators for sweet snacks and dough-based items (Donuts vs. Salads).
* **Protein & Fat:** Key differentiators for meat-based or high-fat items like Burgers and Sushi.

### 3. Preprocessing Steps
To ensure model accuracy, the following steps were taken:
* **Label Encoding:** Categorical variables like `Meal_Type` and `Is_Vegan` were converted into numerical format.
* **Standard Scaling:** Nutritional values (Calories, Sodium, etc.) vary greatly in scale. We applied `StandardScaler` to normalize these features, ensuring the model treats them with equal importance.

### 4. Model Performance
I utilized a **Random Forest Classifier** for this multi-class task. The model achieved high performance across most categories.
* **Precision:** The model showed high accuracy in specifically identifying "Pizza" and "Donut" categories.
* **Recall:** The model successfully captured the majority of instances for each food class despite the imbalanced nature of the dataset.
* **Final Assessment:** The Random Forest algorithm proved robust for this tabular data, effectively handling the non-linear relationships between nutrients and food labels.

