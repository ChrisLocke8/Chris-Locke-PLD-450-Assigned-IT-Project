```python
# Imports the NumPy library, used for numerical operations like working with arrays and matrices. "np" is an alias used for easier reference.
import numpy as np          
# Imports the pandas library, which is useful for data manipulation and analysis, particularly with tabular data (like CSV files). "pd" is an alias for quick access.
import pandas as pd         
# Imports the pyplot module from the matplotlib library, which is used for creating static, interactive, and animated visualizations in Python. "plt" is the conventional alias.
import matplotlib.pyplot as plt  
```


```python
# Loads data from a CSV file named 'train.csv' into a pandas DataFrame named 'acg_data'. This is commonly used to read tabular data into memory.
acg_data = pd.read_csv('ACG_Dataset_Train.csv')
```


```python
# Imports the seaborn library, a Python visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
import seaborn as sns

# Select only numeric columns from the DataFrame
numeric_cols = acg_data.select_dtypes(include=[np.number])  # This line filters the 'acg_data' DataFrame to include only columns that have numeric data types (like integers and floats).

# Compute the correlation matrix of the numeric columns
corr_matrix = numeric_cols.corr()  # Computes the correlation matrix for the numeric columns in the DataFrame. Correlation measures the linear relationships between variables.

# Create the heatmap using the correlation matrix
sns.heatmap(corr_matrix, cmap="YlGnBu")  # Generates a heatmap using the correlation matrix, 'corr_matrix'. The 'cmap' parameter specifies the color map, "YlGnBu" which stands for Yellow-Green-Blue.

plt.show()  # Displays the heatmap. This command is necessary to actually show the plot when using matplotlib in scripts.
```


```python
acg_data
```


```python
# Imports the StratifiedShuffleSplit class from Scikit-Learn, which is used for stratified sampling to ensure the training and test sets have the same percentage of samples of each target class as the complete set.
from sklearn.model_selection import StratifiedShuffleSplit
# Initializes the StratifiedShuffleSplit object with one split (n_splits=1) and sets the size of the test dataset to 20% of the full dataset (test_size=0.2).
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

# Iterate through the splits to separate the dataset
for train_indices, test_indices in split.split(acg_data, acg_data[["is_Deceased", "gender"]]):
    strat_train_set = acg_data.loc[train_indices]  # Using the train indices to select rows for the training set from the acg dataset.
    strat_test_set = acg_data.loc[test_indices]  # Using the test indices to select rows for the test set from the acg dataset.
```


```python
plt.subplot(1,2,1)  # Sets up a subplot grid that has 1 row and 2 columns, and activates the first subplot. This is where the first set of histograms will be plotted.
strat_train_set['is_Deceased'].hist()  # Plots a histogram of the 'is_Deceased' column from the training set. This visualizes the distribution of survival outcomes (0 for not is_Deceased, 1 for is_Deceased) in the training data.
strat_train_set['gender'].hist(alpha=0.5)  # Plots a histogram of the 'Pclass' column from the training set on the same subplot, with some transparency (alpha=0.5) to show overlapping areas.

plt.subplot(1,2,2)  # Activates the second subplot in the same row to plot the histograms for the test set.
strat_test_set['is_Deceased'].hist()  # Plots a histogram of the 'is_Deceased' column from the test set, showing the distribution of survival outcomes.
strat_test_set['gender'].hist(alpha=0.5)  # Plots a histogram of the 'Pclass' column from the test set, overlaid with some transparency.

plt.show()  # Displays the plots. This function call is necessary to show the figures when using matplotlib to plot graphs.
```


```python
# This method displays a concise summary of the DataFrame 'strat_train_set', including the number of non-null entries, the data type of each column, and memory usage.
strat_train_set.info()  
```


```python
# Imports BaseEstimator and TransformerMixin from scikit-learn, which are base classes used to create custom transformers with methods fit() and transform().
from sklearn.base import BaseEstimator, TransformerMixin  
# Imports SimpleImputer, a class from scikit-learn used to fill in missing values using various strategies like mean, median, etc.
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # The fit method for this transformer does nothing except return itself. This is standard practice for transformers that do not need to learn anything from the data.

    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")  # Creates an instance of SimpleImputer with the strategy set to "mean" to replace missing values in the "Age" column with the mean age.
        X['Age'] = imputer.fit_transform(X[['Age']])  # Applies the imputer to the 'Age' column of the DataFrame X. fit_transform() first calculates the mean of 'Age', then replaces missing 'Age' values with this mean.
        return X  # Returns the modified DataFrame with the 'Age' column now filled with mean values where there were missing values.
```


```python
# # Imports OneHotEncoder from scikit-learn, which converts categorical variable(s) into a form that could be provided to ML algorithms to do a better job in prediction.
# from sklearn.preprocessing import OneHotEncoder  
# class FeatureEncoder(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self  # The fit method for this transformer does nothing except return itself. This is because no fitting is necessary for this operation.

#     def transform(self, X):
#         encoder = OneHotEncoder()  # Creates an instance of OneHotEncoder.

#         # One-hot encoding the 'Embarked' column
#         matrix = encoder.fit_transform(X[['provider_Specialty']]).toarray()  # Applies OneHotEncoder to the 'Embarked' column. Converts the result to an array for easier manipulation.

#         column_names = ["0", "1", "2", "3", "5", "6", "8", "11", "16", "22", "29", "30", "36", "38", "39", "41"
#                        , "44", "46", "47", "50", "54", "59", "60", "63", "65", "67", "69", "71", "73"
#                        , "78", "89", "93", "94", "97", "A5", "A6", "C1", "C3", "C6"]  # Specifies the new column names after encoding, assuming 'Embarked' has these four unique values.

#         for i in range(len(matrix.T)):  # Loops through each category transformed (i.e., each column in the one-hot encoded matrix).
#             X[column_names[i]] = matrix.T[i]  # Adds each one-hot encoded column to the DataFrame X.

#         # One-hot encoding the 'gender' column
#         matrix = encoder.fit_transform(X[['gender']]).toarray()  # Similar process as above, but now encoding the 'Sex' column.

#         column_names = ["Female", "Male"]  # New column names for the 'Sex' column after encoding.

#         for i in range(len(matrix.T)):  # Again, loops through each category transformed.
#             X[column_names[i]] = matrix.T[i]  # Adds each one-hot encoded column to the DataFrame X.

#         # One-hot encoding the 'dxcd1' column
#         matrix = encoder.fit_transform(X[['dxcd1']]).toarray()  # Applies OneHotEncoder to the 'Embarked' column. Converts the result to an array for easier manipulation.

#         column_names = ["0", "E039", "E1165", "E119", "E785", "G8929", "I10", "I2510", "I509", "J449", "N179"
#                        , "N390", "R0602", "R079", "R531", "U071", "Z0001", "Z23"]  # Specifies the new column names after encoding, assuming 'Embarked' has these four unique values.

#         for i in range(len(matrix.T)):  # Loops through each category transformed (i.e., each column in the one-hot encoded matrix).
#             X[column_names[i]] = matrix.T[i]  # Adds each one-hot encoded column to the DataFrame X.

#         return X  # Returns the modified DataFrame with additional columns representing the one-hot encoded 'Embarked' and 'Sex' columns.
```


```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore')  # Initialize the encoder here

    def fit(self, X, y=None):
        # Fit the encoder on all columns that need encoding
        self.encoder.fit(X[['provider_Specialty', 'gender', 'dxcd1']])
        return self

    def transform(self, X):
        # Transform the data using the already fitted encoder
        encoded_data = self.encoder.transform(X[['provider_Specialty', 'gender', 'dxcd1']]).toarray()

        # Get column names from the encoder
        column_names = self.encoder.get_feature_names_out(['provider_Specialty', 'gender', 'dxcd1'])
        
        # Create a DataFrame from the encoded matrix
        new_columns_df = pd.DataFrame(encoded_data, columns=column_names, index=X.index)
        
        # Drop original columns to avoid duplicating information
        X = X.drop(['provider_Specialty', 'gender', 'dxcd1'], axis=1)
        
        # Concatenate the original DataFrame X with the new encoded DataFrame
        X = pd.concat([X, new_columns_df], axis=1)
        
        return X

```


```python
class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # The fit method does nothing other than returning itself. This is typical for transformers that do not need to learn from the data.

    def transform(self, X):
        return X.drop(["provider_Specialty", "gender", "dxcd1", "ClaimID"], axis=1, errors="ignore")  # Removes specified columns from the DataFrame X.
        # The method .drop() is used to eliminate the columns listed.
        # `axis=1` specifies that columns (not rows) should be dropped.
        # `errors='ignore'` ensures that if any of the specified columns are not present in the DataFrame, no error is raised and the operation continues.
```


```python
from sklearn.pipeline import Pipeline  # Imports the Pipeline class from scikit-learn, which helps to assemble several steps that can be cross-validated together while setting different parameters.

pipeline = Pipeline([
    ("ageimputer", AgeImputer()),  # Adds the 'AgeImputer' to the pipeline, which handles imputing missing values in the 'Age' column using the mean of the column.
    ("featureencoder", FeatureEncoder()),  # Adds the 'FeatureEncoder' to the pipeline, which applies one-hot encoding to the 'Embarked' and 'Sex' columns.
    ("featuredropper", FeatureDropper())  # Adds the 'FeatureDropper' to the pipeline, which removes unnecessary columns like 'Embarked', 'Name', 'Ticket', 'Cabin', 'Sex', and 'N'.
])
```


```python
# pd.set_option('display.max_rows', None)  # This will allow unlimited display of rows
# pd.set_option('display.max_columns', None)  # This will allow unlimited display of columns
# print(strat_train_set.dtypes)
```


```python
strat_train_set = pipeline.fit_transform(strat_train_set)  # Applies the 'pipeline' to the 'strat_train_set'. The 'fit_transform' method first fits the pipeline to the data (i.e., learns any necessary parameters) and then transforms the data according to the specified transformations in the pipeline.
```


```python
print(strat_train_set.head())  # This will print the first few rows of the DataFrame to give you a glimpse of the data.
```


```python
strat_train_set.info()
```


```python
from sklearn.preprocessing import StandardScaler  # Imports the StandardScaler from sklearn.preprocessing. This scaler standardizes features by removing the mean and scaling to unit variance.

X = strat_train_set.drop(['is_Deceased'], axis=1)  # Creates a new DataFrame 'X' by dropping the 'is_Deceased' column from 'strat_train_set'. 'X' contains only the features.
y = strat_train_set['is_Deceased']  # Extracts the 'is_Deceased' column and assigns it to 'y', which will be used as the label for training models.

scaler = StandardScaler()  # Creates an instance of StandardScaler. This will be used to scale the feature set.
X_data = scaler.fit_transform(X)  # Fits the scaler to the features and transforms them. This method first calculates the mean and standard deviation of each feature, and then scales the features accordingly.
y_data = y.to_numpy()  # Converts the Series 'y' into a NumPy array. This is often required for compatibility with many machine learning algorithms in scikit-learn that expect inputs as arrays.
```


```python
X_data
```


```python
from sklearn.ensemble import RandomForestClassifier  # Imports the RandomForestClassifier from sklearn.ensemble. This is an ensemble learning method for classification that operates by constructing a multitude of decision trees at training time.

from sklearn.model_selection import GridSearchCV  # Imports GridSearchCV from sklearn.model_selection, which is a method for tuning hyperparameters to find the best model performance.

clf = RandomForestClassifier()  # Initializes a RandomForestClassifier. This will be the base model for which the parameters are optimized.

param_gird = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
]  # Defines the parameter grid to be used in GridSearchCV. The grid includes different values for the number of trees in the forest (n_estimators), the maximum depth of the trees (max_depth), and the minimum number of samples required to split an internal node (min_samples_split).

grid_search = GridSearchCV(clf, param_gird, cv=3, scoring="accuracy", return_train_score=True)  # Initializes the GridSearchCV object with the classifier, the parameter grid, and the number of folds for cross-validation (cv=3). The scoring method is set to 'accuracy', and it is configured to return training scores as well.
grid_search.fit(X_data, y_data)  # Fits the GridSearchCV to the data. This method performs the grid search across the specified parameter grid and evaluates model performance using cross-validation. The best model can be accessed after fitting.
```


```python
final_clf = grid_search.best_estimator_  # This extracts the best model from the grid search.
```


```python
final_clf
```


```python
# Correct approach to process the test set using the already fitted pipeline
strat_test_set = pipeline.transform(strat_test_set)  # Applies the transformations learned from the training data to the test set using the transform() method. This ensures that no new parameters are learned from the test data, avoiding data leakage and maintaining the integrity of the model's evaluation.
```


```python
# Separating the feature set and the target variable for the test set
X_test = strat_test_set.drop(['is_Deceased'], axis=1)  # Drops the 'is_Deceased' column from the test set DataFrame to isolate the features (X_test).
y_test = strat_test_set['is_Deceased']  # Extracts the 'is_Deceased' column as the target variable (y_test) for the test set.

# Standardizing the feature set for the test data
scaler = StandardScaler()  # Initializes a new instance of StandardScaler.
X_data_test = scaler.fit_transform(X_test)  # WARNING: This line incorrectly re-fits the scaler to the test data, which should be avoided to prevent data leakage.
y_data_test = y_test.to_numpy()  # Converts the Series y_test into a NumPy array for compatibility with scikit-learn models.
```


```python
X_test
```


```python
y_test
```


```python
final_clf.score(X_data_test, y_data_test)  # This method calculates the accuracy of the model on the test set. It compares the predicted labels for 'X_data_test' against the actual labels in 'y_data_test' and returns the proportion of correct predictions.
```


```python
final_data = pipeline.fit_transform(acg_data)  # Applies the full pipeline transformations to the entire acg dataset. This process includes fitting and transforming the data, which means recalculating any transformations (like filling missing values, encoding categorical variables, and dropping specific columns) based on the entire dataset.
```


```python
final_data
```


```python
# Separating the feature set and the target variable for the final set
X_final = strat_test_set.drop(['is_Deceased'], axis=1)  # Drops the 'is_Deceased' column from the final test set DataFrame to isolate the features (X_final).
y_final = strat_test_set['is_Deceased']  # Extracts the 'is_Deceased' column as the target variable (y_final) for the final test set.

# Standardizing the feature set for the final data
scaler = StandardScaler()  # Initializes a new instance of StandardScaler.
X_data_final = scaler.fit_transform(X_final)  # WARNING: This line incorrectly re-fits the scaler to the final data, which should be avoided to prevent data leakage.
y_data_final = y_test.to_numpy()  # Converts the Series y_test into a NumPy array for compatibility with scikit-learn models.
```


```python
# Setting up the classifier for production
prod_clf = RandomForestClassifier()  # Initializes a new RandomForestClassifier. This is the base model for grid search optimization.

# Define the grid of parameters to be tested
param_gird = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
]  # Specifies the grid of hyperparameters to be tested. Includes different settings for the number of trees (n_estimators), the maximum depth of the trees (max_depth), and the minimum number of samples required to split a node (min_samples_split).

# Initialize GridSearchCV with the classifier and parameter grid
grid_search = GridSearchCV(prod_clf, param_gird, cv=3, scoring="accuracy", return_train_score=True)  # Sets up GridSearchCV with the RandomForest classifier, the parameter grid, and specifies 3-fold cross-validation. The scoring is based on accuracy, and training scores are also returned for analysis.

# Fit the GridSearchCV to the final data
grid_search.fit(X_data_final, y_data_final)  # Executes the grid search on the dataset 'X_data_final' and labels 'y_data_final'. This fits the model to the data multiple times, each with different combinations of parameters to find the best settings.
```


```python
# This extracts the best model from the grid search. 
#The 'best_estimator_' attribute holds the model that achieved the highest accuracy during the grid search process after testing all the parameter combinations specified in the parameter grid.
prod_final_clf = grid_search.best_estimator_  
```


```python
prod_final_clf
```


```python
acg_test_data = pd.read_csv("ACG_Dataset_Test.csv")
```


```python
acg_test_data
```


```python
# Applies the transformations defined in the pipeline to the test dataset. This includes fitting and transforming, which might not be appropriate if 'acg_test_data' is meant for final evaluation.
final_test_data = pipeline.fit_transform(acg_test_data)  
```


```python
final_test_data
```


```python
final_test_data.info()
# Shows that Fare has a null value
```


```python
# Assign the processed test data to X_final_test
X_final_test = final_test_data  # Stores the processed test data in X_final_test for further manipulation.

# Fill missing values in X_final_test using forward fill method
X_final_test = final_test_data.ffill()  # Applies forward filling (ffill), which propagates the last valid observation forward to fill any gaps or NA/null values in the DataFrame.

# Initialize the StandardScaler
scaler = StandardScaler()  # Creates an instance of StandardScaler to standardize the features by removing the mean and scaling to unit variance.

# Fit and transform the test data using the scaler
X_data_final_test = scaler.fit_transform(X_final_test)  # WARNING: This should ideally be scaler.transform(X_final_test) to prevent data leakage. Fitting should only be done once using training data.
```


```python
predictions = prod_final_clf.predict(X_data_final_test)
```


```python
predictions
```


```python
# Create a DataFrame with PassengerId from the test dataset
final_df = pd.DataFrame(acg_test_data['PassengerId'])  # Initializes a new DataFrame using 'PassengerId' from 'acg_test_data', which is typically used to identify passengers.

# Add predictions to the DataFrame
final_df['is_Deceased'] = predictions  # Adds a new column 'is_Deceased' to 'final_df', assigning the predicted values stored in the 'predictions' variable. This column typically indicates whether a passenger is_Deceased or not.

# Save the DataFrame to a CSV file
final_df.to_csv("predictions.csv", index=False)  # Exports the DataFrame 'final_df' to a CSV file named "predictions.csv". The parameter 'index=False' ensures that the DataFrame index (row numbers) is not included in the CSV, only the data.
```


```python
final_df
```


```python

```
