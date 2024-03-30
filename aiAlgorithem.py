import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # For Classification
from sklearn.cluster import KMeans  # For Clustering
from sklearn.linear_model import LinearRegression  # For Regression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Importing the Dataset
csv_file = "C:\Users\Amit\OneDrive\desktop\Kinneret\ai\Ready4School.csv"
dataset = pd.read_csv(csv_file, encoding='utf-8')
# Note: Adjust encoding as needed based on the file's character encoding

# Assuming the dataset includes columns as described; replace 'path_to_your_csv_file.csv' with your actual file path

# Step 3: Taking Care of Missing Data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
numerical_cols = ['Arithmetic', 'Reading', 'Dictation', 'Exercise', 'Comprehension', 
                  'Analogy', 'ArithmeticOps', 'BalloonCount', 'CountNumbers', 
                  'MoreOrLess', 'Triangles']  # Specify correct columns based on your dataset
dataset[numerical_cols] = imputer.fit_transform(dataset[numerical_cols])

# Step 4: Encoding Categorical Data
# Assuming 'Gender' is a categorical variable that needs encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Gender'])], remainder='passthrough')
dataset = np.array(ct.fit_transform(dataset))

# Adjust target variable index as necessary
X = dataset[:, :-1]  # Exclude the target variable
y = dataset[:, -1]   # Target variable

# Step 5: Splitting the Dataset into the Training Set and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 7: Applying Data Mining Techniques
## a. Classification with Logistic Regression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred_class = classifier.predict(X_test)

## b. Clustering with KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
y_kmeans = kmeans.predict(X_test)

## c. Regression with Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_reg = regressor.predict(X_test)

# Evaluation for Classification
print("Classification Report:")
print(classification_report(y_test, y_pred_class))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

# Visualizing Clustering Results (simplified for first two features)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans Clustering')

# Visualizing Regression Results (simplified for first feature)
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], y_test, color='red')
plt.plot(X_test[:, 0], y_pred_reg, color='blue')
plt.title('Linear Regression')
plt.show()