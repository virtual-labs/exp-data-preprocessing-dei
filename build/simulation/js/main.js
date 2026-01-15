/* Main Logic for Experiment Simulation */

/* 
 * Steps Data Configuration 
 */
const stepsData = [
    {
        id: 'import_libraries',
        title: 'Importing Libraries',
        blocks: [
            {
                code: `# Importing Required Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
print("Libraries Imported")`,
                output: `<div class="output-success">Libraries Imported</div>`
            }
        ]
    },
    {
        id: 'reading_data',
        title: 'Loading Dataset',
        blocks: [
            {
                code: `#  Load the Titanic dataset
file_path = "Titanic-Dataset.csv"
data = pd.read_csv(file_path)
print("Dataset loaded successfully")`,
                output: `<div class="output-text">Dataset loaded successfully</div>`
            }
        ]
    },
    {
        id: 'data_analysis',
        title: 'Data Analysis',
        blocks: [
            {
                code: `<div class="output-success"># Display the first 5 rows of the dataset</div>
data.head()`,
                output: `<table class="data-table">
  <thead>
    <tr>
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td><td>1</td><td>0</td><td>3</td><td>Braund, Mr. Owen Harris</td><td>male</td><td>22.0</td><td>1</td><td>0</td><td>A/5 21171</td><td>7.2500</td><td>NaN</td><td>S</td>
    </tr>
    <tr>
      <td>1</td><td>2</td><td>1</td><td>1</td><td>Cumings, Mrs. John Bradley (Florence Briggs Th...)</td><td>female</td><td>38.0</td><td>1</td><td>0</td><td>PC 17599</td><td>71.2833</td><td>C85</td><td>C</td>
    </tr>
    <tr>
      <td>2</td><td>3</td><td>1</td><td>3</td><td>Heikkinen, Miss. Laina</td><td>female</td><td>26.0</td><td>0</td><td>0</td><td>STON/O2. 3101282</td><td>7.9250</td><td>NaN</td><td>S</td>
    </tr>
    <tr>
      <td>3</td><td>4</td><td>1</td><td>1</td><td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td><td>female</td><td>35.0</td><td>1</td><td>0</td><td>113803</td><td>53.1000</td><td>C123</td><td>S</td>
    </tr>
    <tr>
      <td>4</td><td>5</td><td>0</td><td>3</td><td>Allen, Mr. William Henry</td><td>male</td><td>35.0</td><td>0</td><td>0</td><td>373450</td><td>8.0500</td><td>NaN</td><td>S</td>
    </tr>
  </tbody>
</table>`

            },
            {
                code: `<div class="output-success"># Display dataset structure and data types</div>
            data.info()`,
                output: `<div class="output-text" style="white-space: pre; font-family: monospace;">&lt;class 'pandas.core.frame.DataFrame'&gt;
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB</div>`
            },
            {
                code: `<div class="output-success" ># Show the number of rows and columns in the dataset</div>
    data.shape`,
                output: `<div class="output-text" > (891, 12)</div> `
            },
            {
                code: `<div class="output-success" ># Count occurrences of each class in the Survived column</div>
    data['Survived'].value_counts()`,
                output: `<div class="output-text" > 0    549</div><div class="output-text">1    342</div><div class="output-text">Name: Survived, dtype: int64</div>`
            },

            {
                code: `<div class="output-success" ># Count frequency of each Sex value</div>
    data['Sex'].value_counts()`,
                output: `<div class="output-text" > <strong>Sex</strong></div>
<div class="output-text">male      577</div>
<div class="output-text">female    314</div>
<div class="output-text">Name: Sex, dtype: int64</div>`
            },
        ]
    },
    {
        id: 'data_preprocessing',
        title: 'Data Preprocessing',
        blocks: [
            {
                code: `<div class="output-success"># Visualize and display missing values in each column (Handling Missing Values)</div>
missing_count = data.isna().sum()
print("Missing values count per column:")
display(missing_count.to_frame(name="Missing Count"))`,
                output: `<div class="output-text" style="font-family: monospace; color: #5FA8E4; font-weight: bold; margin-bottom: 5px;">Missing values count per column:</div>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 8px 15px; text-align: left;"></th>
      <th style="padding: 8px 15px; text-align: right;">Missing Count</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">PassengerId</td><td style="padding: 4px 15px; text-align: right;">0</td></tr>
    <tr><td style="padding: 4px 15px;">Survived</td><td style="padding: 4px 15px; text-align: right;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">Pclass</td><td style="padding: 4px 15px; text-align: right;">0</td></tr>
    <tr><td style="padding: 4px 15px;">Name</td><td style="padding: 4px 15px; text-align: right;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">Sex</td><td style="padding: 4px 15px; text-align: right;">0</td></tr>
    <tr><td style="padding: 4px 15px;">Age</td><td style="padding: 4px 15px; text-align: right;">177</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">SibSp</td><td style="padding: 4px 15px; text-align: right;">0</td></tr>
    <tr><td style="padding: 4px 15px;">Parch</td><td style="padding: 4px 15px; text-align: right;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">Ticket</td><td style="padding: 4px 15px; text-align: right;">0</td></tr>
    <tr><td style="padding: 4px 15px;">Fare</td><td style="padding: 4px 15px; text-align: right;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">Cabin</td><td style="padding: 4px 15px; text-align: right;">687</td></tr>
    <tr><td style="padding: 4px 15px;">Embarked</td><td style="padding: 4px 15px; text-align: right;">2</td></tr>
  </tbody>
</table>`
            },
            {
                code: `<div class="output-success"># Plot the percentage of missing values for each column</div>
missing_percentage = (df.isna().mean() * 100).sort_values(ascending=False)

plt.figure()
missing_percentage.plot(kind="bar")
plt.title("Percentage of Missing Values per Column")
plt.ylabel("Percentage (%)")
plt.xlabel("Columns")
plt.xticks(rotation=45)
plt.show()`,
                output: `<div style="text-align:left; padding:10px;">
    <img src="./images/percent_missing_values_per_column.png" alt="Percentage of Missing Values per Column" style="max-height: 350px; border: 1px solid #ddd; border-radius: 4px;">
</div>`
            },
            {
                code: `<div class="output-success"># Fill missing Age values using mean</div>
temp_df = df.copy()
temp_df["Age"] = temp_df["Age"].fillna(temp_df["Age"].mean())

print("Age filled using Mean")
print("Remaining missing values in Age:", temp_df["Age"].isna().sum())
display(temp_df[["Age"]].head())`,
                output: `<div class="output-text" style="color: #5FA8E4; font-weight: bold; margin-bottom: 5px;">Age filled using Mean</div>
<div class="output-text">Remaining missing values in Age: 0</div>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 4px 15px;"></th>
      <th style="padding: 4px 15px;">Age</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">22.0</td></tr>
    <tr><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">38.0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">26.0</td></tr>
    <tr><td style="padding: 4px 15px;">3</td><td style="padding: 4px 15px;">35.0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">4</td><td style="padding: 4px 15px;">35.0</td></tr>
  </tbody>
</table>`
            },
            {
                code: `<div class="output-success"># Remaining missing value in age after filling</div>
df["Age"] = df["Age"].fillna(df["Age"].mean())
print("Remaining missing values in Age:", df["Age"].isna().sum())`,
                output: `<div class="output-text">Remaining missing values in Age: 0</div>`
            },
            {
                code: `<div class="output-success"># Find the most frequent value in Embarked</div>
most_frequent_embarked = df["Embarked"].mode()[0]
print("Most frequent Embarked value:", most_frequent_embarked)`,
                output: `<div class="output-text">Most frequent Embarked value: S</div>`
            },
            {
                code: `<div class="output-success"># Fill missing Embarked values with the most frequent value</div>
df_embarked = df.copy()
df_embarked["Embarked"] = df_embarked["Embarked"].fillna(most_frequent_embarked)

print("Remaining missing values in Embarked:", df_embarked["Embarked"].isna().sum())
display(df_embarked[["Embarked"]].head())`,
                output: `<div class="output-text">Remaining missing values in Embarked: 0</div>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
       <th style="padding: 4px 15px;"></th>
       <th style="padding: 4px 15px;">Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">S</td></tr>
    <tr><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">C</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">S</td></tr>
    <tr><td style="padding: 4px 15px;">3</td><td style="padding: 4px 15px;">S</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">4</td><td style="padding: 4px 15px;">S</td></tr>
  </tbody>
</table>`
            },
            {
                code: `<div class="output-success"># as there are many null values we Drop the Cabin column</div>
df = df.drop(columns=["Cabin"], errors="ignore")
df.head()`,
                output: `<table class="data-table">
  <thead>
    <tr>
      <th></th><th>PassengerId</th><th>Survived</th><th>Pclass</th><th>Name</th><th>Sex</th><th>Age</th><th>SibSp</th><th>Parch</th><th>Ticket</th><th>Fare</th><th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>1</td><td>0</td><td>3</td><td>Braund, Mr. Owen Harris</td><td>male</td><td>22.0</td><td>1</td><td>0</td><td>A/5 21171</td><td>7.2500</td><td>S</td></tr>
    <tr><td>1</td><td>2</td><td>1</td><td>1</td><td>Cumings, Mrs. John Bradley (Florence Briggs Th...)</td><td>female</td><td>38.0</td><td>1</td><td>0</td><td>PC 17599</td><td>71.2833</td><td>C</td></tr>
    <tr><td>2</td><td>3</td><td>1</td><td>3</td><td>Heikkinen, Miss. Laina</td><td>female</td><td>26.0</td><td>0</td><td>0</td><td>STON/O2. 3101282</td><td>7.9250</td><td>S</td></tr>
    <tr><td>3</td><td>4</td><td>1</td><td>1</td><td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td><td>female</td><td>35.0</td><td>1</td><td>0</td><td>113803</td><td>53.1000</td><td>S</td></tr>
    <tr><td>4</td><td>5</td><td>0</td><td>3</td><td>Allen, Mr. William Henry</td><td>male</td><td>35.0</td><td>0</td><td>0</td><td>373450</td><td>8.0500</td><td>S</td></tr>
  </tbody>
</table>`
            },
            {
                code: `<div class="output-success"># Create a cleaned dataset and confirm there are no missing values</div>
df_clean = df.copy()
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])
print("Total missing values after handling:", df_clean.isna().sum().sum())`,
                output: `<div class="output-text">Total missing values after handling: 0</div>`
            },
            {
                code: `<div class="output-success"># Select numerical columns to apply Min-Max scaling</div>
num_cols = ["Age", "Fare"]
df_num = df_clean[num_cols].copy()`,
                output: `<div class="output-success">MinMaxScaler imported and numerical columns selected</div>`
            },
            {
                code: `<div class="output-success"># Scale numerical columns to the 0-1 range using MinMaxScaler</div>
scaler = MinMaxScaler()

df_scaled = df_num.copy()
df_scaled[num_cols] = scaler.fit_transform(df_num[num_cols])`,
                output: `<div class="output-success">Numerical columns balanced and scaled between 0 and 1</div>`
            },
            {
                code: `<div class="output-success"># Compare Age distribution before and after Min-Max scaling</div>
plt.figure()
plt.hist(df_num["Age"], bins=30, density=True)
plt.title("Age Distribution (Before Scaling)")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()

plt.figure()
plt.hist(df_scaled["Age"], bins=30, density=True)
plt.title("Age Distribution (After Min-Max Scaling)")
plt.xlabel("Scaled Age")
plt.ylabel("Density")
plt.show()`,
                output: `<div style="display: flex; flex-direction: row; gap: 20px; justify-content: flex-start; align-items: flex-start;">
    <div style="text-align:left;">
        <img src="./images/age_distribution_before_scaling.png" alt="Age Distribution Before Scaling" style="max-height: 350px; border: 1px solid #ddd; border-radius: 4px;">
    </div>
    <div style="text-align:left;">
        <img src="./images/age_distribution_after_min_max_scaling.png" alt="Age Distribution After Scaling" style="max-height: 350px; border: 1px solid #ddd; border-radius: 4px;">
    </div>
</div>`
            },
            {
                code: `<div class="output-success"># Compare Fare distribution before and after Min-Max scaling</div>
plt.figure()
plt.hist(df_num["Fare"], bins=30)
plt.title("Fare Distribution (Before Scaling)")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(df_scaled["Fare"], bins=30)
plt.title("Fare Distribution (After Min-Max Scaling)")
plt.xlabel("Scaled Fare")
plt.ylabel("Frequency")
plt.show()`,
                output: `<div style="display: flex; flex-direction: row; gap: 20px; justify-content: flex-start; align-items: flex-start;">
    <div style="text-align:left;">
        <img src="./images/fare_distribution_before_scaling.png" alt="Fare Distribution Before Scaling" style="max-height: 350px; border: 1px solid #ddd; border-radius: 4px;">
    </div>
    <div style="text-align:left;">
        <img src="./images/fare_distribution_after_min-max_scaling.png" alt="Fare Distribution After Scaling" style="max-height: 350px; border: 1px solid #ddd; border-radius: 4px;">
    </div>
</div>`
            },
            {
                code: `<div class="output-success"># List and display categorical columns</div>
categorical_cols = ["Sex", "Embarked", "Pclass"]

print("Categorical Columns:")
print(categorical_cols)`,
                output: `<div class="output-text">Categorical Columns:</div>
<div class="output-text">['Sex', 'Embarked', 'Pclass']</div>`
            },
            {
                code: `<div class="output-success"># Apply binary encoding to the 'Sex' column</div>
df_sex_encoded = df_clean.copy()
df_sex_encoded["Sex"] = df_sex_encoded["Sex"].map({"male": 0, "female": 1})

print("Sex column after Binary Encoding:")
display(df_sex_encoded[["Sex"]].head())`,
                output: `<div class="output-text" style="color: #5FA8E4; font-weight: bold; margin-bottom: 5px;">Sex column after Binary Encoding:</div>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 4px 15px;"></th>
      <th style="padding: 4px 15px;">Sex</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">0</td></tr>
    <tr><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">1</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">1</td></tr>
    <tr><td style="padding: 4px 15px;">3</td><td style="padding: 4px 15px;">1</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">4</td><td style="padding: 4px 15px;">0</td></tr>
  </tbody>
</table>`
            },
            {
                code: `<div class="output-success"># One-Hot Encode the 'Embarked' column</div>
df_embarked_encoded = df_clean.copy()
df_embarked_encoded = pd.get_dummies(df_embarked_encoded, columns=["Embarked"], prefix="Embarked")

print("Embarked column after One-Hot Encoding:")
display(df_embarked_encoded.filter(like="Embarked").head())`,
                output: `<div class="output-text" style="color: #5FA8E4; font-weight: bold; margin-bottom: 5px;">Embarked column after One-Hot Encoding:</div>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 4px 15px;"></th>
      <th style="padding: 4px 15px;">Embarked_C</th>
      <th style="padding: 4px 15px;">Embarked_Q</th>
      <th style="padding: 4px 15px;">Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">True</td></tr>
    <tr><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">True</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">True</td></tr>
    <tr><td style="padding: 4px 15px;">3</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">True</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">4</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">True</td></tr>
  </tbody>
</table>`
            },
            {
                code: `<div class="output-success"># One-Hot Encode the 'Pclass' column</div>
df_pclass_encoded = df_clean.copy()
df_pclass_encoded = pd.get_dummies(df_pclass_encoded, columns=["Pclass"], prefix="Pclass")

print("Pclass column after One-Hot Encoding:")
display(df_pclass_encoded.filter(like="Pclass").head())`,
                output: `<div class="output-text" style="color: #5FA8E4; font-weight: bold; margin-bottom: 5px;">Pclass column after One-Hot Encoding:</div>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 4px 15px;"></th>
      <th style="padding: 4px 15px;">Pclass_1</th>
      <th style="padding: 4px 15px;">Pclass_2</th>
      <th style="padding: 4px 15px;">Pclass_3</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">True</td></tr>
    <tr><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">True</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">True</td></tr>
    <tr><td style="padding: 4px 15px;">3</td><td style="padding: 4px 15px;">True</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">4</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">False</td><td style="padding: 4px 15px;">True</td></tr>
  </tbody>
</table>`
            }
        ]
    },

    {
        id: 'visualization',
        title: 'Visualization',
        blocks: [
            {
                code: `<div class="output-success"># Plot the count of passengers who survived vs. did not survive</div>
plt.figure()
df_clean["Survived"].value_counts().sort_index().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()`,
                output: `<img src="./images/survival_count.png" alt="Survival Count" style="max-height: 350px; border-radius: 4px;">`
            },
            {
                code: `<div class="output-success"># Plot the distribution of passenger ages</div>
plt.figure()
plt.hist(df_clean["Age"], bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()`,
                output: `<img src="./images/age_distribution.png" alt="Age Distribution" style="max-height: 350px; border-radius: 4px;">`
            },
            {
                code: `<div class="output-success"># Plot a grouped bar chart of survival counts by sex</div>
plt.figure()
df_clean.groupby(["Sex", "Survived"]).size().unstack().plot(kind="bar")
plt.title("Survival vs Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()`,
                output: `<img src="./images/survival_vs_sex.png" alt="Survival vs Sex" style="max-height: 350px; border-radius: 4px;">`
            },
            {
                code: `<div class="output-success"># Plot a grouped bar chart of survival counts by passenger class</div>
plt.figure()
df_clean.groupby(["Pclass", "Survived"]).size().unstack().plot(kind="bar")
plt.title("Survival vs Pclass")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()`,
                output: `<img src="./images/survival_vs_Pclass.png" alt="Survival vs Pclass" style="max-height: 350px; border-radius: 4px;">`
            },
            {
                code: `<div class="output-success"># Compare Fare distribution by survival status using a box plot</div>
plt.figure()
df_clean.boxplot(column="Fare", by="Survived")
plt.title("Fare vs Survival")
plt.suptitle("")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare")
plt.show()`,
                output: `<img src="./images/Fare_vs_Survival.png" alt="Fare vs Survival" style="max-height: 350px; border-radius: 4px;">`
            },

        ]
    },
    {
        id: 'feature_engineering',
        title: 'Feature Engineering',
        blocks: [
            {
                code: `<div class="output-success"># Create a new feature 'FamilySize' as SibSp + Parch + 1</div>
df_family = df_clean.copy()
df_family["FamilySize"] = df_family["SibSp"] + df_family["Parch"] + 1

print("FamilySize feature created:")
display(df_family[["SibSp", "Parch", "FamilySize"]].head())`,
                output: `<div class="output-text" style="color: #5FA8E4; font-weight: bold; margin-bottom: 5px;">FamilySize feature created:</div>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 4px 15px;"></th>
      <th style="padding: 4px 15px;">SibSp</th>
      <th style="padding: 4px 15px;">Parch</th>
      <th style="padding: 4px 15px;">FamilySize</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">2</td></tr>
    <tr><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">2</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">1</td></tr>
    <tr><td style="padding: 4px 15px;">3</td><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">2</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">4</td><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">1</td></tr>
  </tbody>
</table>`
            },
            {
                code: `<div class="output-success"># Create a new feature 'IsAlone' where 1 indicates traveling alone</div>
df_isalone = df_family.copy()
df_isalone["IsAlone"] = (df_isalone["FamilySize"] == 1).astype(int)

print("IsAlone feature created:")
display(df_isalone[["FamilySize", "IsAlone"]].head())`,
                output: `<div class="output-text" style="color: #5FA8E4; font-weight: bold; margin-bottom: 5px;">IsAlone feature created:</div>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 4px 15px;"></th>
      <th style="padding: 4px 15px;">FamilySize</th>
      <th style="padding: 4px 15px;">IsAlone</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">0</td></tr>
    <tr><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">1</td></tr>
    <tr><td style="padding: 4px 15px;">3</td><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">4</td><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">1</td></tr>
  </tbody>
</table>`
            },
            {
                code: `<div class="output-success"># Add 'FamilySize' and 'IsAlone' features to the dataset</div>
df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
df_clean["IsAlone"] = (df_clean["FamilySize"] == 1).astype(int)

display(df_clean[["FamilySize", "IsAlone"]].head())`,
                output: `<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 4px 15px;"></th>
      <th style="padding: 4px 15px;">FamilySize</th>
      <th style="padding: 4px 15px;">IsAlone</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">0</td></tr>
    <tr><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">1</td></tr>
    <tr><td style="padding: 4px 15px;">3</td><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">4</td><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">1</td></tr>
  </tbody>
</table>`
            },
            {
                code: `<div class="output-success"># Final cleaned dataset</div>
df_clean.describe()`,
                output: `<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 4px 15px;"></th>
      <th style="padding: 4px 15px;">PassengerId</th>
      <th style="padding: 4px 15px;">Survived</th>
      <th style="padding: 4px 15px;">Pclass</th>
      <th style="padding: 4px 15px;">Age</th>
      <th style="padding: 4px 15px;">SibSp</th>
      <th style="padding: 4px 15px;">Parch</th>
      <th style="padding: 4px 15px;">Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">count</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td></tr>
    <tr><td style="padding: 4px 15px;">mean</td><td style="padding: 4px 15px;">446.000000</td><td style="padding: 4px 15px;">0.383838</td><td style="padding: 4px 15px;">2.308642</td><td style="padding: 4px 15px;">29.699118</td><td style="padding: 4px 15px;">0.523008</td><td style="padding: 4px 15px;">0.381594</td><td style="padding: 4px 15px;">32.204208</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">std</td><td style="padding: 4px 15px;">257.353842</td><td style="padding: 4px 15px;">0.486592</td><td style="padding: 4px 15px;">0.836071</td><td style="padding: 4px 15px;">13.002015</td><td style="padding: 4px 15px;">1.102743</td><td style="padding: 4px 15px;">0.806057</td><td style="padding: 4px 15px;">49.693429</td></tr>
    <tr><td style="padding: 4px 15px;">min</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">0.420000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">0.000000</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">25%</td><td style="padding: 4px 15px;">223.500000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">2.000000</td><td style="padding: 4px 15px;">22.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">7.910400</td></tr>
    <tr><td style="padding: 4px 15px;">50%</td><td style="padding: 4px 15px;">446.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">3.000000</td><td style="padding: 4px 15px;">29.699118</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">14.454200</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">75%</td><td style="padding: 4px 15px;">668.500000</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">3.000000</td><td style="padding: 4px 15px;">35.000000</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">31.000000</td></tr>
    <tr><td style="padding: 4px 15px;">max</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">3.000000</td><td style="padding: 4px 15px;">80.000000</td><td style="padding: 4px 15px;">8.000000</td><td style="padding: 4px 15px;">6.000000</td><td style="padding: 4px 15px;">512.329200</td></tr>
  </tbody>
</table>`
            }
        ]
    }
];

// State Management
let STATE = {
    stepIndex: 0,
    subStepIndex: 0,
    stepsStatus: stepsData.map(() => ({ unlocked: false, completed: false, partial: false }))
};

// Initial State: First step unlocked
STATE.stepsStatus[0].unlocked = true;

// DOM Elements
const stepsContainer = document.getElementById('stepsContainer');
const codeDisplay = document.getElementById('codeDisplay');
const outputDisplay = document.getElementById('outputDisplay');
const outputContent = document.getElementById('outputDisplay'); // Wrapper reuse
const bottomPane = document.querySelector('.bottom-pane');
const runBtn = document.getElementById('runBtn');

// Initialize UI
function init() {
    renderSidebar();
    loadStep(0);
}

// Render Sidebar with Color Logic
function renderSidebar() {
    stepsContainer.innerHTML = '';

    stepsData.forEach((step, index) => {
        const status = STATE.stepsStatus[index];
        const btn = document.createElement('button');
        btn.classList.add('step-btn');
        btn.innerText = step.title;

        // Label Logic
        let label = `${index + 1}. ${step.title} `;
        if (status.completed) label = `✓ ${step.title} `;
        btn.innerText = label;

        // Styling & Interaction Logic
        if (status.unlocked) {
            if (status.completed) {
                btn.classList.add('completed');
            } else if (status.partial) {
                btn.classList.add('in-progress');
            } else {
                // Default unlocked state (Blue via CSS)
            }

            btn.disabled = false;
            btn.style.cursor = 'pointer';

            // Mark current active
            if (index === STATE.stepIndex) {
                btn.classList.add('active');
            }

            // Allow click to load/revisit
            btn.onclick = () => {
                loadStep(index);
            };
        } else {
            // Locked -> Grey
            btn.classList.add('disabled');
            btn.innerText = label;
            btn.disabled = true;
            btn.onclick = (e) => e.preventDefault();
        }

        stepsContainer.appendChild(btn);
    });

    // Loading Spinner (Hidden by default)
    const loader = document.createElement('div');
    loader.className = 'loading-spinner';
    loader.innerText = 'Loading...';
    loader.style.width = '100%';
    loader.style.textAlign = 'center';
    loader.style.marginTop = '20px';
    loader.style.display = 'none';
    stepsContainer.appendChild(loader);

    // Add Restart Button at the end
    const restartBtn = document.createElement('button');
    restartBtn.classList.add('step-btn');
    restartBtn.innerText = "Restart Experiment";
    restartBtn.style.backgroundColor = "#333";
    restartBtn.style.textAlign = 'center';
    restartBtn.style.marginTop = "auto";
    restartBtn.style.color = "white";
    restartBtn.onclick = restartExperiment;
    stepsContainer.appendChild(restartBtn);

    // Add Download Button below Restart
    const downloadBtn = document.createElement('button');
    downloadBtn.classList.add('step-btn');
    downloadBtn.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align: middle;">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="7 10 12 15 17 10"></polyline>
      <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
    Download Experiment
  `;
    downloadBtn.style.backgroundColor = "#F57C2A"; // Orange (#F57C2A)
    downloadBtn.style.textAlign = 'center';
    downloadBtn.style.marginTop = "10px";
    downloadBtn.style.color = "white";
    downloadBtn.onclick = downloadPDF;
    stepsContainer.appendChild(downloadBtn);
}


function loadStep(index) {
    STATE.stepIndex = index;
    STATE.subStepIndex = 0; // Fix: Always reset sub-step when loading a main step
    renderSidebar();
    updateUI();
}

function updateUI() {
    const step = stepsData[STATE.stepIndex];
    if (STATE.subStepIndex >= step.blocks.length) STATE.subStepIndex = 0;

    const block = step.blocks[STATE.subStepIndex];

    // Extract and Update Comment Header
    const commentMatch = block.code.match(/#\s*([^<\n\r]*)/); // Extract comment until tag or newline
    const codeHeaderBar = document.getElementById('codeHeaderBar');
    if (commentMatch) {
        codeHeaderBar.innerText = "# " + commentMatch[1].trim();
        codeHeaderBar.style.display = 'block';
    } else {
        codeHeaderBar.style.display = 'none';
    }

    // Update Code (Remove all HTML tags and then extract code)
    const codeWithoutTags = block.code.replace(/<[^>]*>/g, '');
    const codeWithoutComment = codeWithoutTags.replace(/#\s*.*/, '').trim();
    codeDisplay.innerHTML = highlightCode(codeWithoutComment);

    // Reset Output
    bottomPane.classList.remove('active-output');
    // Reset any inline styles added by completion message
    bottomPane.style.display = '';
    bottomPane.style.flexDirection = '';
    bottomPane.style.justifyContent = '';
    bottomPane.style.alignItems = '';

    outputContent.innerHTML = '<div class="placeholder-text">Click the Run button to execute...</div>';

    // Reset Button State (Simple & Safe)
    runBtn.style.display = 'flex'; // Fix: Ensure button is visible after restart
    runBtn.classList.remove('completed');
    runBtn.style.backgroundColor = '#F57C2A'; // Orange (#F57C2A)
    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
    runBtn.disabled = false;

    // ALWAYS reset onclick to standard runStep
    runBtn.onclick = runStep;
}

function runStep() {
    const step = stepsData[STATE.stepIndex];
    const block = step.blocks[STATE.subStepIndex];

    // 1. Loading State
    outputContent.innerHTML = '<div class="loading-spinner">Running code...</div>';
    runBtn.disabled = true;

    // 2. Simulated Delay (2 seconds)
    setTimeout(() => {
        // 3. Show Output
        outputContent.innerHTML = block.output;
        bottomPane.classList.add('active-output');

        // 4. Update Button State to Checkmark (Success)
        runBtn.classList.add('completed');
        runBtn.style.backgroundColor = '#A6CE63'; // Green (#A6CE63)
        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';

        // Mark partial progress
        STATE.stepsStatus[STATE.stepIndex].partial = true;
        renderSidebar();

        // Check if this is the Random Prediction block
        if (document.getElementById('randomPredTableBody')) {
            window.generateRandomPrediction && window.generateRandomPrediction();
        }

        // 5. Handle Next Logic
        const hasNextBlock = STATE.subStepIndex < step.blocks.length - 1;

        if (hasNextBlock) {
            // Wait 1s then change button to "Next"
            setTimeout(() => {
                runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                runBtn.style.backgroundColor = '#5FA8E4'; // Orange
                runBtn.disabled = false;

                // Switch handler to Next
                runBtn.onclick = nextSubStep;
            }, 500);

        } else {
            // Step Fully Completed
            STATE.stepsStatus[STATE.stepIndex].completed = true;
            renderSidebar(); // Update Current Step to Green Immediately

            // Unlock next step logic
            if (STATE.stepIndex < stepsData.length - 1) {
                STATE.stepsStatus[STATE.stepIndex + 1].unlocked = true;
                renderSidebar(); // Update Next Step to Red Immediately

                // Manual Next Step Arrow Button
                setTimeout(() => {
                    // Change button to Blue Arrow for Next Step
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.style.backgroundColor = '#5FA8E4'; // Blue (#5FA8E4)
                    runBtn.disabled = false;

                    // Logic to go to next MAIN step
                    runBtn.onclick = function () {
                        loadStep(STATE.stepIndex + 1);
                    };
                }, 500);
            } else {
                // End of Experiment - Show "Next" (Finish) Button
                renderSidebar();
                setTimeout(() => {
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.style.backgroundColor = '#72b2f7ff'; // Orange
                    runBtn.disabled = false;
                    runBtn.onclick = showCompletionMessage;
                }, 500);
            }
        }

    }, 500);
}

function nextSubStep() {
    STATE.subStepIndex++;
    updateUI(); // This will reset button to Red/Run for the new block
}

function restartExperiment() {
    // Reset State
    STATE.stepIndex = 0;
    STATE.subStepIndex = 0;
    STATE.stepsStatus = stepsData.map(() => ({ unlocked: false, completed: false, partial: false }));
    STATE.stepsStatus[0].unlocked = true;

    init();
}

function highlightCode(code) {
    return code
        .replace(/import /g, '<span class="kw">import </span>')
        .replace(/from /g, '<span class="kw">from </span>')
        .replace(/print/g, '<span class="func">print</span>')
        .replace(/def /g, '<span class="kw">def </span>')
        .replace(/return /g, '<span class="kw">return </span>');
}

// Global scope for HTML callbacks
window.updateSigmoidPlot = function () {
    const featureSelect = document.getElementById('featureSelect');
    if (!featureSelect) return; // Guard

    const feature = featureSelect.value;
    const container = document.getElementById('sigmoidPlot');

    // Use static image matching the feature name
    // Default to Age if feature is just "Feature" or empty, but here we read value
    // Ensure the image fits comfortably without massive white borders
    container.innerHTML = `< img src = "./images/${feature}.png" alt = "Sigmoid of ${feature}" style = "max-height:300px; border:none; display:block;" > `;
};

// Global scope for Random Prediction Table
window.generateRandomPrediction = function () {
    const tbody = document.getElementById('randomPredTableBody');
    if (!tbody) return;

    const samples = [
        { index: 101, pclass: 3, sex: 0, age: 22, sibsp: 1, parch: 0, fare: 7.25, survived: 0, pred: 0, prob: 0.12 },
        { index: 102, pclass: 1, sex: 1, age: 38, sibsp: 1, parch: 0, fare: 71.28, survived: 1, pred: 1, prob: 0.94 },
        { index: 103, pclass: 3, sex: 1, age: 26, sibsp: 0, parch: 0, fare: 7.92, survived: 1, pred: 1, prob: 0.65 },
        { index: 104, pclass: 1, sex: 1, age: 35, sibsp: 1, parch: 0, fare: 53.10, survived: 1, pred: 1, prob: 0.89 },
        { index: 105, pclass: 3, sex: 0, age: 35, sibsp: 0, parch: 0, fare: 8.05, survived: 0, pred: 0, prob: 0.08 }
    ];

    tbody.innerHTML = '';

    samples.forEach(s => {
        const tr = document.createElement('tr');
        tr.style.transition = 'background 0.2s';
        tr.onmouseover = () => { if (!tr.classList.contains('selected-row')) tr.style.background = '#e3f2fd'; };
        tr.onmouseout = () => { if (!tr.classList.contains('selected-row')) tr.style.background = 'white'; };
        tr.onclick = () => window.showPredictionResult(s, tr);

        tr.innerHTML = `
            <td><strong>${s.index}</strong></td>
            <td>${s.pclass}</td>
            <td>${s.sex}</td>
            <td>${s.age}</td>
            <td>${s.sibsp}</td>
            <td>${s.parch}</td>
            <td>${s.fare.toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
    });
};

window.showPredictionResult = function (s, tr) {
    try {
        const resDiv = document.getElementById('randomPredResult');
        if (!resDiv) return;

        const rows = document.querySelectorAll('#randomPredTableBody tr');
        rows.forEach(r => {
            r.style.backgroundColor = 'white';
            r.classList.remove('selected-row');
        });

        tr.style.backgroundColor = '#a5d6a7';
        tr.classList.add('selected-row');
        resDiv.style.display = 'block';

        const center = (str, w) => {
            const val = (str === undefined || str === null) ? '' : str.toString();
            if (val.length >= w) return val;
            const pad = ' '.repeat(Math.max(0, w - val.length));
            return val + pad;
        };

        const out = `
------ Random Sample Test ------
Feature    Value
-------    -----
Index      ${s.index}
Pclass     ${s.pclass}
Sex        ${s.sex}
Age        ${s.age}
SibSp      ${s.sibsp}
Parch      ${s.parch}
Fare       ${s.fare.toFixed(2)}

Actual Survival: ${s.survived}
Predicted Survival: ${s.pred}
Probability: ${s.prob.toFixed(4)}
        `.trim();

        resDiv.innerText = out;

    } catch (e) {
        console.error("Error in showPredictionResult:", e);
    }
};

// Global scope for Confusion Matrix Animation (Simplified for Image)
window.animateConfusionMatrix = function () {
    // No animation needed for static image, but keeping function to prevent errors if called
};

// Completion Message
function showCompletionMessage() {
    outputContent.innerHTML = ''; // Clear output content
    bottomPane.classList.add('active-output');
    bottomPane.style.display = 'flex';
    bottomPane.style.flexDirection = 'column';
    bottomPane.style.justifyContent = 'center';
    bottomPane.style.alignItems = 'center';

    const msgHTML = `
    <div style = "text-align: center; animation: fadeIn 1s ease;" >
      <h1 style="color: #2a9d8f; font-size: 2.5rem; margin-bottom: 20px;">Experiment Completed! ✔️</h1>
      <p style="font-size: 1.5rem; color: #333;">You have completed Data Preprocessing and Feature Engineering successfully!</p>
      <button onclick="location.reload()" style="margin-top: 30px; padding: 15px 30px; background-color: #f7a072; color: white; border: none; border-radius: 10px; font-size: 1.2rem; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">Restart Experiment</button>
    </div>
    `;
    outputContent.innerHTML = msgHTML;
    // Hide run button or make it inactive
    runBtn.style.display = 'none';
}

// PDF Download Logic
function downloadPDF() {
    const link = document.createElement('a');
    link.href = './Exp-Data-Preprocessing.pdf';
    link.download = 'Exp-Data-Preprocessing.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}



// Init
function init() {
    renderSidebar();
    loadStep(0);

    // Attach Download Listener
    const downloadBtn = document.querySelector('.download-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadPDF);
    }
}

init();
