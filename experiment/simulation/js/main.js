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
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

import ipywidgets as widgets
from IPython.display import display
from scipy.stats import zscore 
from IPython.display import display, HTML, clear_output
from matplotlib.pyplot import summer
print("Libraries imported")`,
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
      },

      {
        code: `# Convert Age to integer
df_raw = pd.read_csv("Titanic-Dataset.csv")
df_clean = df_raw.copy()
df_clean["Age"] = df_clean["Age"].round().astype("Int64")
print(df_clean["Age"])`,
        output: `<div class="output-text" style="white-space: pre; font-family: monospace;">0       22
1       38
2       26
3       35
4       35
      ... 
886     27
887     19
888   &lt;NA&gt;
889     26
890     32
Name: Age, Length: 891, dtype: Int64</div>`
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
        code: `# Statistical Summary / Descriptors
display(HTML("<h3 style='text-align:center;'>Statistical Summary of Dataset</h3>"))
summary = df.describe()
display(summary)`,
        output: `<h3 style='text-align:left;'>Statistical Summary of Dataset</h3>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace; text-align: right;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black; font-weight: bold;">
      <th style="padding: 4px 15px; text-align: left;"></th>
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
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px; text-align: left; font-weight: bold;">count</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">714.000000</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">891.000000</td></tr>
    <tr><td style="padding: 4px 15px; text-align: left; font-weight: bold;">mean</td><td style="padding: 4px 15px;">446.000000</td><td style="padding: 4px 15px;">0.383838</td><td style="padding: 4px 15px;">2.308642</td><td style="padding: 4px 15px;">29.699118</td><td style="padding: 4px 15px;">0.523008</td><td style="padding: 4px 15px;">0.381594</td><td style="padding: 4px 15px;">32.204208</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px; text-align: left; font-weight: bold;">std</td><td style="padding: 4px 15px;">257.353842</td><td style="padding: 4px 15px;">0.486592</td><td style="padding: 4px 15px;">0.836071</td><td style="padding: 4px 15px;">14.526497</td><td style="padding: 4px 15px;">1.102743</td><td style="padding: 4px 15px;">0.806057</td><td style="padding: 4px 15px;">49.693429</td></tr>
    <tr><td style="padding: 4px 15px; text-align: left; font-weight: bold;">min</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">0.420000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">0.000000</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px; text-align: left; font-weight: bold;">25%</td><td style="padding: 4px 15px;">223.500000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">2.000000</td><td style="padding: 4px 15px;">20.125000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">7.910400</td></tr>
    <tr><td style="padding: 4px 15px; text-align: left; font-weight: bold;">50%</td><td style="padding: 4px 15px;">446.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">3.000000</td><td style="padding: 4px 15px;">28.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">14.454200</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px; text-align: left; font-weight: bold;">75%</td><td style="padding: 4px 15px;">668.500000</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">3.000000</td><td style="padding: 4px 15px;">38.000000</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">0.000000</td><td style="padding: 4px 15px;">31.000000</td></tr>
    <tr><td style="padding: 4px 15px; text-align: left; font-weight: bold;">max</td><td style="padding: 4px 15px;">891.000000</td><td style="padding: 4px 15px;">1.000000</td><td style="padding: 4px 15px;">3.000000</td><td style="padding: 4px 15px;">80.000000</td><td style="padding: 4px 15px;">8.000000</td><td style="padding: 4px 15px;">6.000000</td><td style="padding: 4px 15px;">512.329200</td></tr>
  </tbody>
</table>`
      },

    ]
  },
  {
    id: 'missing_values_imputation',
    title: 'Missing Values Imputation',
    blocks: [
      {
        code: `<div class="output-success"># (HANDLING MISSING VALUES) Visualize and display missing values in each column</div>
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
        code: `# Interactive widget to handle missing values in 'Age' using Mean/Median/Mode/Constant with preview and apply option

# INTERACTIVE NUMERIC MISSING VALUE IMPUTATION

feature = "Age"
display(HTML("<h3 style='text-align:center;'>INTERACTIVE NUMERIC MISSING VALUE IMPUTATION</h3>"))

method_dropdown = widgets.Dropdown(
    options=['Mean','Median','Mode','Constant'],
    description='Method'
)
constant_value = widgets.IntText(
    value=0,
    description='Constant'
)
apply_button = widgets.Button(
    description="Impute the Missing Values",
    button_style='success',
    layout=widgets.Layout(width='420px', height='40px')
)

output = widgets.Output()
imputation_done = False

# IMPORTANT: keep a copy of original data
df_preview = df_clean.copy()
nan_rows_fixed = df_preview[df_preview[feature].isna()].sample(5)
non_nan_rows_fixed = df_preview[df_preview[feature].notna()].sample(5)

sample_index = pd.concat([nan_rows_fixed, non_nan_rows_fixed]).index

def interactive_impute(change=None):

    global imputation_done

    with output:
        clear_output()
        temp_df = df_clean.copy()
        missing_before = temp_df[feature].isna().sum()
        print("The numeric feature 'Age' contains 177 missing values.")

        method = method_dropdown.value

        if method == "Mean":
            value = int(round(temp_df[feature].mean()))
            label = "Mean"

        elif method == "Median":
            value = int(round(temp_df[feature].median()))
            label = "Median"

        elif method == "Mode":
            value = int(temp_df[feature].mode()[0])
            label = "Mode"

        else:
            value = constant_value.value
            label = "Constant"

        after_series = temp_df[feature].fillna(value)

        result = pd.DataFrame({
            "Before_Imputation_Age": df_preview[feature],
            "After_Imputation_Age": after_series
        })

        # ALWAYS take NaN rows from original preview copy
        sample_table = result.loc[sample_index]
        sample_table.index = [''] * len(sample_table)

        # keep After column empty before clicking button
        if not imputation_done:
            sample_table["After_Imputation_Age"] = ""

        display(HTML(f"""
        <div style="width:420px; text-align:center; margin-bottom:5px;">
        
        <div style="color:#1d3557; font-size:15px; margin-bottom:3px;">
            {missing_before} NaN missing values will be replaced with {label} {value} 
        </div>

        <div style="color:red; font-size:28px;">
            ↑
        </div>

        </div>
        """))

        display(apply_button)

        print("\\nRandom 10 Sample Showing Imputation Effect")

        display(sample_table)


def apply_imputation(b):

    global imputation_done
    method = method_dropdown.value

    if method == "Mean":
        value = int(round(df_clean[feature].mean()))
        label = "Mean"

    elif method == "Median":
        value = int(round(df_clean[feature].median()))
        label = "Median"

    elif method == "Mode":
        value = int(df_clean[feature].mode()[0])
        label = "Mode"

    else:
        value = constant_value.value
        label = "Constant"

    df_clean[feature] = df_clean[feature].fillna(value)

    imputation_done = True

    with output:
        print(f"\\nImputation finalized. Missing values in Age have been permanently filled using {label} method.")

    interactive_impute()


def reset_preview(change=None):
    global imputation_done
    imputation_done = False
    interactive_impute()

method_dropdown.observe(reset_preview, names='value')
constant_value.observe(reset_preview, names='value')

apply_button.on_click(apply_imputation)

display(method_dropdown, constant_value, output)

interactive_impute()
`,
        output: `<div id="imputationInteractiveUI" style="display: inline-block; font-family: sans-serif; padding: 15px; background: white; text-align: left; border-radius: 4px;">
            <div style="font-weight: bold; font-size: 14px; text-align: center; margin-bottom: 15px;">
                INTERACTIVE NUMERIC MISSING VALUE IMPUTATION
            </div>
            <div style="display: flex; flex-direction: column; gap: 5px; margin-left: 20px; font-size: 13px;">
                <div style="display: flex; align-items: center;">
                    <label style="margin-right: 5px; width: 50px; text-align: right;">Method</label>
                    <select id="simMethodDropdown" style="width: 150px; padding: 2px;" onchange="window.updateSimPreview()">
                        <option value="Mean">Mean</option>
                        <option value="Median">Median</option>
                        <option value="Mode">Mode</option>
                        <option value="Constant">Constant</option>
                    </select>
                </div>
                <div id="simConstantDiv" style="display: none; align-items: center;">
                    <label style="margin-right: 5px; width: 50px; text-align: right;">Constant</label>
                    <input type="number" id="simConstantInput" value="0" style="width: 142px; padding: 2px;" onchange="window.updateSimPreview()">
                </div>
            </div>
            <div id="simPreviewArea" style="text-align: left; width: 100%; margin-top: 10px;"></div>
        </div>`
      },
      {
        code: `# Interactive imputation of categorical missing values (Cabin & Embarked) using most frequent value with preview and apply buttons


# FILLING CATEGORICAL MISSING VALUES WITH THE MOST FREQUENT VALUE

display(HTML("""
<h3 style="text-align:center;">
FILLING CATEGORICAL MISSING VALUES WITH THE MOST FREQUENT VALUE
</h3>
"""))


# CREATE CLEAN COPY

df_clean = df.copy()

# FIND MOST FREQUENT VALUES

most_frequent_embarked = df["Embarked"].mode()[0]
most_frequent_cabin = df["Cabin"].mode()[0]

# OUTPUT AREA

output = widgets.Output()

# BUTTONS

cabin_button = widgets.Button(
    description="Impute Missing Values for Cabin",
    button_style='success',
    layout=widgets.Layout(width='320px', height='40px')
)

embarked_button = widgets.Button(
    description="Impute Missing Values for Embarked",
    button_style='success',
    layout=widgets.Layout(width='320px', height='40px')
)

# FIXED SAMPLE TABLES (NO RESHUFFLE)

cabin_sample = df_clean[["Cabin"]].sample(6).copy()

embarked_nan = df_clean[df_clean["Embarked"].isna()][["Embarked"]]
embarked_non_nan = df_clean[df_clean["Embarked"].notna()][["Embarked"]].sample(4)

embarked_sample = pd.concat([embarked_nan, embarked_non_nan]).copy()

# STATE MEMORY

cabin_done = False
embarked_done = False

# DISPLAY FUNCTION

def display_tables():

    with output:

        clear_output()

        print("Missing values in Embarked:", df_clean["Embarked"].isna().sum())
        print("Missing values in Cabin:", df_clean["Cabin"].isna().sum())

        print("\\nMost frequent value in Embarked:", most_frequent_embarked)
        print("Most frequent value in Cabin:", most_frequent_cabin)

        print("\\nConcept Explanation:")

        print("""
Categorical missing values are commonly filled using the
most frequent category (also called the mode).

Reason:
Replacing missing values with the most frequent category
preserves the distribution of the dataset and avoids introducing
new artificial categories.

In this dataset:
Embarked missing values will be replaced with the most frequent port.
Cabin missing values will be replaced with the most frequent cabin value.

Once the button is pressed, the missing values will be filled
and the dataset will be updated.
""")

        # ---------------- CABIN ----------------

        display(cabin_button)

        cabin_table = cabin_sample.copy()

        if cabin_done:
            cabin_table["After_Cabin"] = cabin_table["Cabin"].fillna(most_frequent_cabin)
        else:
            cabin_table["After_Cabin"] = ""

        cabin_table.columns = ["Before_Cabin","After_Cabin"]
        cabin_table.index = [""] * len(cabin_table)

        print("\\n6 Random Samples Showing Cabin Imputation Effect")
        display(cabin_table)

        # ---------------- EMBARKED ----------------

        display(embarked_button)

        embarked_table = embarked_sample.copy()

        if embarked_done:
            embarked_table["After_Embarked"] = embarked_table["Embarked"].fillna(most_frequent_embarked)
        else:
            embarked_table["After_Embarked"] = ""

        embarked_table.columns = ["Before_Embarked","After_Embarked"]
        embarked_table.index = [""] * len(embarked_table)

        print("\\n6 Random Samples Showing Embarked Imputation Effect")
        display(embarked_table)

# BUTTON FUNCTIONS

def fill_cabin(b):
    global cabin_done
    df_clean["Cabin"] = df_clean["Cabin"].fillna(most_frequent_cabin)
    cabin_done = True
    display_tables()


def fill_embarked(b):
    global embarked_done
    df_clean["Embarked"] = df_clean["Embarked"].fillna(most_frequent_embarked)
    embarked_done = True
    display_tables()

# BUTTON EVENTS

cabin_button.on_click(fill_cabin)
embarked_button.on_click(fill_embarked)

# DISPLAY

display(output)
display_tables()`,
        output: function () {
          setTimeout(window.updateCatSimPreview, 50);
          return `<div id="catImputationInteractiveUI" style="display: inline-block; font-family: monospace; padding: 15px; background: white; text-align: left; border-radius: 4px; width: auto; min-width: 500px;">
            <div style="font-weight: bold; font-family: sans-serif; font-size: 14px; text-align: right; margin-bottom: 5px;">
                FILLING CATEGORICAL MISSING VALUES WITH THE MOST FREQUENT VALUE
            </div>
            <div id="catSimPreviewArea" style="text-align: left; width: 100%; margin-top: 10px;"></div>
        </div>`;
        }
      }
    ]
  },
  {
    id: 'encoding_categorical_variables',
    title: 'Encoding Categorical Variables',
    blocks: [

      {
        code: `# Interactive encoding of categorical features (Sex, Embarked, Cabin, Pclass) using Label, One-Hot, and Ordinal encoding with preview and apply option


# ENCODING CATEGORICAL VARIABLES

display(HTML("<h3 style='text-align:center;'>Encoding Categorical Variables</h3>"))


# EXTRACT DECK LETTER FROM CABIN


if "Cabin" in df_clean.columns:
    df_clean["Cabin"] = df_clean["Cabin"].astype(str).str[0]

# TRACK ENCODED FEATURES

encoded_features = set()
before_snapshots = {}

# ENCODING METHOD DROPDOWN

encoding_method = widgets.Dropdown(
    options=["Label Encoding","One-Hot Encoding","Ordinal Encoding"],
    description="Encoding:"
)

# FEATURE MAP

feature_map = {
    "Label Encoding": ["Sex"],
    "One-Hot Encoding": ["Embarked","Cabin"],
    "Ordinal Encoding": ["Pclass"]
}

# FEATURE DROPDOWN

encoding_feature = widgets.Dropdown(
    options=[],
    description="Feature:"
)

# OUTPUT

preview_output = widgets.Output()

# BUTTON

encode_button = widgets.Button(
    description="Encode",
    button_style='success',
    layout=widgets.Layout(width='220px', height='45px')
)

# UPDATE FEATURE DROPDOWN

def update_feature_dropdown(change=None):

    method = encoding_method.value

    available = [f for f in feature_map[method] if f not in encoded_features]

    if len(available) == 0:
        encoding_feature.options = []
        encoding_feature.value = None
        encode_button.disabled = True
        encode_button.description = "No Feature Available"
        return

    encoding_feature.options = available
    encoding_feature.value = available[0]

    update_button()

encoding_method.observe(update_feature_dropdown, names="value")

# UPDATE BUTTON

def update_button():

    feature = encoding_feature.value

    if feature is None:
        encode_button.disabled = True
        encode_button.button_style = ''
        encode_button.description = "Encode"
        return

    encode_button.description = f"Encode {feature}"

    if feature in encoded_features:
        encode_button.disabled = True
        encode_button.button_style = ''
    else:
        encode_button.disabled = False
        encode_button.button_style = 'success'

encoding_feature.observe(lambda change: update_button(), names="value")

update_feature_dropdown()

# PREVIEW FUNCTION

def encode_toggle(method, feature):

    with preview_output:

        clear_output()

        if feature is None:
            return

        temp = df_clean.copy()

        if feature not in before_snapshots and feature in temp.columns:
            before_snapshots[feature] = temp[[feature]].head().copy()

        if feature in before_snapshots:
            before = before_snapshots[feature]
        else:
            return

        print("Selected Encoding Method:", method)
        print("Selected Feature:", feature)

        if feature in temp.columns:
            temp[feature] = temp[feature].astype(str)

        # LABEL ENCODING
        if method == "Label Encoding":

            print("\\nLabel Encoding:")
            print("""Label Encoding converts categorical values into numbers.

For the feature 'Sex', the two categories are male and female.

Mapping used in this dataset:
female → 0
male → 1

The numbers 0 and 1 act only as identifiers so machine learning
models can process the categorical feature.""")

            le = LabelEncoder()
            temp[feature] = le.fit_transform(temp[feature])

            after = temp[[feature]].head()

        # ONE HOT ENCODING
        elif method == "One-Hot Encoding":

            print("\\nOne-Hot Encoding:")
            print("One-Hot Encoding creates separate binary columns for each category.")

            if feature == "Embarked":

                print("""
For 'Embarked', passengers boarded from three ports:
C (Cherbourg), Q (Queenstown), and S (Southampton).

Therefore three columns are created:
Embarked_C, Embarked_Q, Embarked_S.

Each row contains 1 in the column representing the category
and 0 in the remaining columns.""")

            if feature == "Cabin":

                print("""
For 'Cabin', the deck letter is extracted first.
Possible decks include A, B, C, D, E, F, G and T.

One column is created for each deck such as Cabin_A, Cabin_B etc.
A value of 1 indicates the passenger belongs to that deck.""")

            if feature in temp.columns:
                after = pd.get_dummies(temp[[feature]], prefix=feature).head()
            else:
                after = pd.get_dummies(before, prefix=feature)

            if feature == "Embarked":
                expected_cols = ["Embarked_C","Embarked_Q","Embarked_S"]
                after = after.reindex(columns=expected_cols, fill_value='False')

            if feature == "Cabin":
                expected_cols = ["Cabin_A","Cabin_B","Cabin_C","Cabin_D",
                                 "Cabin_E","Cabin_F","Cabin_G","Cabin_T"]
                after = after.reindex(columns=expected_cols, fill_value='False')

        # ORDINAL ENCODING
        else:

            print("\\nOrdinal Encoding:")
            print("""Ordinal Encoding is used when categories have a natural ranking.

In the Titanic dataset, 'Pclass' represents passenger class:
1 = First Class, 2 = Second Class, 3 = Third Class.

The encoder assigns increasing integers:

1 → 0
2 → 1
3 → 2""")

            enc = OrdinalEncoder()
            temp[[feature]] = enc.fit_transform(temp[[feature]]).astype(int)

            after = temp[[feature]].head()

        before = before.reset_index(drop=True)
        after = after.reset_index(drop=True)

        if feature in encoded_features:
            after_display = after
        else:
            after_display = pd.DataFrame(
                [[""] * after.shape[1]] * after.shape[0],
                columns=after.columns
            )

        display(HTML(f"""
        <div style="display:flex; gap:40px;">
            <div>
                <h4>Dataset Before Encoding</h4>
                {before.to_html(index=False)}
            </div>
            <div>
                <h4>Dataset After Encoding</h4>
                {after_display.to_html(index=False)}
            </div>
        </div>
        """))

# FINALIZE ENCODING

def finalize_encoding(b):

    global df_clean

    feature = encoding_feature.value
    method = encoding_method.value

    if feature not in df_clean.columns:
        return

    if method == "Label Encoding":

        le = LabelEncoder()
        df_clean[feature] = le.fit_transform(df_clean[feature].astype(str))

    elif method == "One-Hot Encoding":

        encoded = pd.get_dummies(df_clean[feature], prefix=feature)
        df_clean = pd.concat([df_clean.drop(columns=[feature]), encoded], axis=1)

    else:

        enc = OrdinalEncoder()
        df_clean[[feature]] = enc.fit_transform(df_clean[[feature]]).astype(int)

    encoded_features.add(feature)

    encode_button.disabled = True
    encode_button.button_style = ''

    encode_toggle(method, feature)

# BUTTON CALLBACK

encode_button.on_click(finalize_encoding)

# DISPLAY

widgets.interact(
    encode_toggle,
    method=encoding_method,
    feature=encoding_feature
)

display(preview_output)
display(encode_button)`,
        output: function () {
          if (!window.encodedFeatures) {
            window.encodedFeatures = new Set();
            window.featureMap = {
              "Label Encoding": ["Sex"],
              "One-Hot Encoding": ["Embarked", "Cabin"],
              "Ordinal Encoding": ["Pclass"]
            };
          }
          setTimeout(() => {
            if (document.getElementById("encMethodDropdown")) {
              document.getElementById("encMethodDropdown").value = "Label Encoding";
              window.onEncMethodChange();
            } else {
              window.updateEncodingPreview();
            }
          }, 50);
          return `<div id="encodingInteractiveUI" style="display: inline-block; font-family: sans-serif; padding: 15px; background: white; text-align: left; border-radius: 4px; width: auto; min-width: 650px;">
                <div style="font-weight: bold; font-family: sans-serif; font-size: 15px; text-align: left; margin-bottom: 20px;">
                    Encoding Categorical Variables
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; margin-left: 20px; font-size: 13px; font-family: sans-serif;">
                    <div style="display: flex; align-items: center;">
                        <label style="margin-right: 5px; width: 60px; text-align: right;">Encoding:</label>
                        <select id="encMethodDropdown" style="width: 150px; padding: 2px;" onchange="window.onEncMethodChange()">
                            <option value="Label Encoding" selected>Label Encoding</option>
                            <option value="One-Hot Encoding">One-Hot Encoding</option>
                            <option value="Ordinal Encoding">Ordinal Encoding</option>
                        </select>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <label style="margin-right: 5px; width: 60px; text-align: right;">Feature:</label>
                        <select id="encFeatureDropdown" style="width: 150px; padding: 2px;" onchange="window.updateEncodingPreview()">
                            <option value="Sex">Sex</option>
                        </select>
                    </div>
                </div>
                <div id="encodingPreviewArea" style="text-align: left; width: 100%; margin-top: 20px;"></div>
            </div>`;
        }
      }
    ]
  },
  {
    id: 'normalization',
    title: 'Normalization',
    blocks: [
      {
        code: `# Interactive scaling of numerical features (Age, Fare) using Standard, MinMax, and Robust scalers with explanation, visualization, and apply option

# INTERACTIVE FEATURE SCALING

display(HTML("<h3 style='text-align:center;'>INTERACTIVE FEATURE SCALING</h3>"))

# WIDGETS

scaled_features = set()

scale_feature = widgets.Dropdown(
    options=["Age","Fare"],
    description="Feature:"
)

scale_method = widgets.Dropdown(
    options=["StandardScaler","MinMaxScaler","RobustScaler"],
    description="Scaler:"
)

scale_button = widgets.Button(
    description="Scale Age",
    button_style='success'
)

output = widgets.Output()

# BUTTON TEXT UPDATE

def update_button(change=None):

    feature = scale_feature.value

    # If already scaled → grey disabled
    if feature in scaled_features:
        scale_button.description = f"{feature} Already Scaled"
        scale_button.disabled = True
        scale_button.button_style = ''
        scale_button.style.button_color = "#BDBDBD"
        return

    # If not scaled → enable button again
    scale_button.disabled = False
    scale_button.style.button_color = None

    if feature == "Age":
        scale_button.description = "Scale Age"
        scale_button.button_style = "success"

    else:
        scale_button.description = "Scale Fare"
        scale_button.button_style = "success"

scale_feature.observe(update_button, names="value")

# GET SCALER

def get_scaler():

    if scale_method.value == "StandardScaler":
        return StandardScaler()

    elif scale_method.value == "MinMaxScaler":
        return MinMaxScaler()

    else:
        return RobustScaler()

# PREVIEW (EXPLANATION ONLY)

def preview(change=None):

    with output:

        clear_output()

        scaler_name = scale_method.value

        print("\\nConcept Explanation:")
        print("Feature scaling transforms numerical values so that different features operate on comparable ranges. This prevents machine learning algorithms from being biased toward features with larger magnitudes.")

        if scaler_name == "StandardScaler":

            print("""
StandardScaler (Standardization)

This method transforms the feature so that the resulting values
have a mean of 0 and a standard deviation of 1.

Formula:
z = (x − μ) / σ

Where:
x = original value
μ = mean
σ = standard deviation
""")

        elif scaler_name == "MinMaxScaler":

            print("""
MinMaxScaler (Normalization)

This method rescales the data to a fixed range, usually between 0 and 1.

Formula:
x' = (x − xmin) / (xmax − xmin)
""")

        else:

            print("""
RobustScaler

This method scales the data using the median and the interquartile range.

Formula:
x' = (x − median) / IQR
""")

# RUN SCALING (BUTTON)

def run_scaling(b):

    with output:

        clear_output()

        feature = scale_feature.value
        scaler_name = scale_method.value

        # SHOW EXPLANATION AGAIN AFTER BUTTON PRESS
        print("\\nConcept Explanation:")
        print("Feature scaling transforms numerical values so that different features operate on comparable ranges. This prevents machine learning algorithms from being biased toward features with larger magnitudes.")

        if scaler_name == "StandardScaler":

            print("""
StandardScaler (Standardization)

This method transforms the feature so that the resulting values
have a mean of 0 and a standard deviation of 1.

Formula:
z = (x − μ) / σ

Where:
x = original value
μ = mean
σ = standard deviation
""")

        elif scaler_name == "MinMaxScaler":

            print("""
MinMaxScaler (Normalization)

This method rescales the data to a fixed range, usually between 0 and 1.

Formula:
x' = (x − xmin) / (xmax − xmin)
""")

        else:

            print("""
RobustScaler

This method scales the data using the median and the interquartile range.

Formula:
x' = (x − median) / IQR
""")

        data = df_clean[feature].dropna()
        reshaped = data.values.reshape(-1,1)

        sc = get_scaler()
        scaled = sc.fit_transform(reshaped).flatten()

        # Save scaled values
        df_clean[f"Scaled_{feature}"] = np.nan
        df_clean.loc[data.index, f"Scaled_{feature}"] = scaled
        scaled_features.add(feature)
        
        scale_button.disabled = True
        scale_button.button_style = ''
        scale_button.style.button_color = "#BDBDBD"

        # GRAPH

        fig, ax = plt.subplots(1,2,figsize=(10,4))

        sns.histplot(data, bins=20, ax=ax[0], kde=True)
        ax[0].set_title("Before Scaling")

        sns.histplot(scaled, bins=20, ax=ax[1], kde=True)
        ax[1].set_title(f"After {scaler_name}")

        plt.show()

        # TABLE

        table = pd.DataFrame({
            f"Actual_{feature}": data.values.astype(int),
            f"Scaled_{feature}": scaled.round(3)
        })

        sample_table = table.sample(30)
        sample_table.index = [""] * len(sample_table)

        display(HTML("<b>Sample Values Showing Scaling Effect</b>"))

        display(HTML(f"""
        <div style="height:220px; overflow-y:auto; border:1px solid #ccc;">
        {sample_table.to_html(index=False)}
        </div>
        """))

        print(f"\\nScaled {feature} values saved in df_clean as column: Scaled_{feature}")

# EVENTS

scale_method.observe(preview, names="value")
scale_feature.observe(preview, names="value")

scale_button.on_click(run_scaling)

# DISPLAY

display(scale_feature)
display(scale_method)
display(scale_button)
display(output)

preview()`,
        output: function () {
          if (!window.scaledFeatures) {
            window.scaledFeatures = new Set();
            window.scaleFeature = "Age";
            window.scaleMethod = "StandardScaler";
          }
          setTimeout(() => {
            if (document.getElementById("scaleMethodDropdown")) {
              document.getElementById("scaleMethodDropdown").value = "StandardScaler";
              document.getElementById("scaleFeatureDropdown").value = "Age";
              window.onScaleFeatureChange();
            } else {
              window.updateScalingPreview();
            }
          }, 50);
          return `<div id="scalingInteractiveUI" style="display: inline-block; font-family: sans-serif; padding: 15px; background: white; text-align: left; border-radius: 4px; width: auto; min-width: 650px;">
                <div style="font-weight: bold; font-family: sans-serif; font-size: 14px; text-align: center; margin-bottom: 20px;">
                    INTERACTIVE FEATURE SCALING
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-size: 13px; font-family: sans-serif; margin-bottom: 5px;">
                    <div style="display: flex; align-items: center; justify-content: flex-start;">
                        <label style="margin-right: 5px; width: 60px; text-align: right;">Feature:</label>
                        <select id="scaleFeatureDropdown" style="width: 150px; padding: 2px;" onchange="window.onScaleFeatureChange()">
                            <option value="Age">Age</option>
                            <option value="Fare">Fare</option>
                        </select>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: flex-start;">
                        <label style="margin-right: 5px; width: 60px; text-align: right;">Scaler:</label>
                        <select id="scaleMethodDropdown" style="width: 150px; padding: 2px;" onchange="window.onScaleMethodChange()">
                            <option value="StandardScaler">StandardScaler</option>
                            <option value="MinMaxScaler">MinMaxScaler</option>
                            <option value="RobustScaler">RobustScaler</option>
                        </select>
                    </div>
                </div>
                <div id="scalingPreviewArea" style="text-align: left; width: 100%; margin-left: 20px;"></div>
            </div>`;
        }
      },

    ]
  },

  {
    id: 'outlier_detection',
    title: 'Outlier Detection',
    blocks: [
      {
        code: `# Interactive outlier detection for numerical features (Age, Fare) using IQR and Z-Score methods with explanation and visualization

# INTERACTIVE OUTLIER DETECTION

display(HTML("<h3 style='text-align:center;'>INTERACTIVE OUTLIER DETECTION</h3>"))

# WIDGETS

outlier_feature = widgets.Dropdown(
    options=["Age","Fare"],
    description="Feature:"
)

outlier_method = widgets.Dropdown(
    options=["IQR","Z-Score"],
    description="Method:"
)


# Output area (prevents widgets from disappearing)
output = widgets.Output()

# FUNCTION

def detect_outlier(feature, method):

    with output:
        clear_output(wait=True)

        data = df_clean[feature].dropna()

        # IQR METHOD

        if method == "IQR":

            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            if feature == "Fare":
                multiplier = 4.5
            else:
                multiplier = 2.3

            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR

            outliers = data[(data < lower) | (data > upper)]

            explanation = f"""
IQR (Interquartile Range) Method

The IQR method detects outliers by examining the spread
of the middle 50% of the dataset.

Formula:
IQR = Q3 − Q1

Where:
Q1 = 25th percentile
Q3 = 75th percentile

Outlier limits:

Lower Bound = Q1 − {multiplier} × IQR
Upper Bound = Q3 + {multiplier} × IQR

Traditionally the multiplier used is 1.5 × IQR. However,
datasets such as the Titanic Fare distribution are highly
skewed. Using 1.5 may label many moderate values as outliers.

Therefore a relaxed multiplier ({multiplier} × IQR) is used
to focus on more extreme deviations and reduce excessive
outlier detection.
"""

        # Z-SCORE METHOD

        else:

            z_scores = np.abs(zscore(data))
            outliers = data[z_scores > 3]

            explanation = """
Z-Score Method

The Z-Score method determines how far a data point lies
from the mean of the dataset in terms of standard deviations.

Formula:
z = (x − μ) / σ

Where:
x = data value
μ = mean of dataset
σ = standard deviation

Rule used:
|z| > 3 → Outlier

This means the value is more than three standard deviations
away from the mean.

This approach works best when the data distribution is
approximately normal.
"""

        # RESULTS

        print("Selected Feature:", feature)
        print("Detection Method:", method)
        print("Total Data Points:", len(data))
        print("Number of Outliers Detected:", len(outliers))

        # CONCEPT EXPLANATION (MOVED UP)

        print("\\nConcept Explanation:")
        print(explanation)

        # VISUALIZATION

        plt.figure(figsize=(8,4))

        sns.boxplot(x=data)

        if len(outliers) > 0:
            plt.scatter(
                outliers,
                [0]*len(outliers),
                color="red",
                label="Detected Outliers"
            )
            plt.legend()

        plt.title(f"Outlier Detection for {feature} using {method}")
        plt.show()

# INTERACTIVE CONTROL

widgets.interact(
    detect_outlier,
    feature=outlier_feature,
    method=outlier_method
)

display(output)`,
        output: function () {
          setTimeout(() => {
            if (document.getElementById("outlierMethodDropdown")) {
              document.getElementById("outlierMethodDropdown").value = "IQR";
              document.getElementById("outlierFeatureDropdown").value = "Age";
              window.onOutlierChange();
            } else {
              window.updateOutlierPreview();
            }
          }, 50);
          return `<div id="outlierInteractiveUI" style="display: inline-block; font-family: sans-serif; padding: 15px; background: white; text-align: left; border-radius: 4px; width: auto; min-width: 650px;">
                <div style="font-weight: bold; font-family: sans-serif; font-size: 14px; text-align: center; margin-bottom: 20px;">
                    INTERACTIVE OUTLIER DETECTION
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-size: 13px; font-family: sans-serif; margin-bottom: 5px;">
                    <div style="display: flex; align-items: center; justify-content: flex-start;">
                        <label style="margin-right: 5px; width: 60px; text-align: right;">Feature:</label>
                        <select id="outlierFeatureDropdown" style="width: 150px; padding: 2px;" onchange="window.onOutlierChange()">
                            <option value="Age">Age</option>
                            <option value="Fare">Fare</option>
                        </select>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: flex-start;">
                        <label style="margin-right: 5px; width: 60px; text-align: right;">Method:</label>
                        <select id="outlierMethodDropdown" style="width: 150px; padding: 2px;" onchange="window.onOutlierChange()">
                            <option value="IQR">IQR</option>
                            <option value="Z-Score">Z-Score</option>
                        </select>
                    </div>
                </div>
                <div id="outlierPreviewArea" style="text-align: left; width: 100%; margin-top: 10px;"></div>
            </div>`;
        }
      }
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
display(df_isalone[["IsAlone"]].head())`,
        output: `<div class="output-text" style="color: #5FA8E4; font-weight: bold; margin-bottom: 5px;">IsAlone feature created:</div>
<table class="data-table" style="width: auto; background-color: #2b2b2b; color: #fff; border-collapse: collapse; font-family: monospace;">
  <thead>
    <tr style="border-bottom: 1px solid #444; background-color: #fdfdfd; color: black;">
      <th style="padding: 4px 15px;"></th>
      <th style="padding: 4px 15px;">IsAlone</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">0</td><td style="padding: 4px 15px;">0</td></tr>
    <tr><td style="padding: 4px 15px;">1</td><td style="padding: 4px 15px;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">2</td><td style="padding: 4px 15px;">1</td></tr>
    <tr><td style="padding: 4px 15px;">3</td><td style="padding: 4px 15px;">0</td></tr>
    <tr style="background-color: #3c3f41;"><td style="padding: 4px 15px;">4</td><td style="padding: 4px 15px;">1</td></tr>
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

    ]
  },

  {
    id: 'visualization',
    title: 'Visualization',
    blocks: [
      {
        code: `<div class="output-success"># (UNIVARIATE VISUALIZATION) Plot the count of passengers who survived vs. did not survive</div>
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
        code: `<div class="output-success"># (BIVARIATE VISUALIZATION) Plot a grouped bar chart of survival counts by sex</div>
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

      {
        code: `# Displays side-by-side comparison of raw vs processed dataset (first 10 rows) for quick visualization of changes
# BEFORE VS AFTER DATA COMPARISON

def compare_datasets():

    print("Raw Dataset vs Processed Dataset (first 10 samples)")

    raw = df_raw.head(10).copy()
    processed = df_clean.head(10).copy()

    # Ensure Age is integer
    if "Age" in raw.columns:
        raw["Age"] = raw["Age"].round().astype("Int64")

    if "Age" in processed.columns:
        processed["Age"] = processed["Age"].round().astype("Int64")

    display_html(
        f"""
        &lt;div style="display:flex; gap:40px; align-items:flex-start;"&gt;

            &lt;div style="background:#ffe5b4; padding:15px; border-radius:10px; width:50%;"&gt;
                &lt;h3 style="text-align:center"&gt;Raw Dataset&lt;/h3&gt;
                &lt;div style="overflow-x:auto;"&gt;]
                    {raw.to_html(index=False)}
                &lt;/div&gt;
            &lt;/div&gt;

            &lt;div style="background:#d8f3dc; padding:15px; border-radius:10px; width:50%;"&gt;
                &lt;h3 style="text-align:center"&gt;Processed Dataset&lt;/h3&gt;
                &lt;div style="overflow-x:auto;"&gt;
                    {processed.to_html(index=False)}
                &lt;/div&gt;
            &lt;/div&gt;

        &lt;/div&gt;
        """,
        raw=True
    )

compare_datasets()`,
        output: `<div class="output-text" style="font-family: monospace; font-size: 13px; font-weight: bold; margin-bottom: 5px;">Raw Dataset vs Processed Dataset (first 10 samples)</div>
        <div style="display:flex; gap:40px; align-items:flex-start; font-family: sans-serif; font-size: 12px;">
            <div style="background:#ffe5b4; padding:15px; border-radius:4px; width:45%; max-width: 50%;">
                <h3 style="text-align:center; margin-top: 0; margin-bottom: 10px; font-size: 16px;">Raw Dataset</h3>
                <div style="overflow-x:auto;">
                    <table style="width: 100%; border-collapse: collapse; text-align: right; background: white;">
                        <thead>
                            <tr style="border-bottom: 1px solid #aaa;">
                                <th style="padding: 4px 10px;">PassengerId</th><th style="padding: 4px 10px;">Survived</th><th style="padding: 4px 10px;">Pclass</th><th style="padding: 4px 10px;">Name</th><th style="padding: 4px 10px;">Sex</th><th style="padding: 4px 10px;">Age</th><th style="padding: 4px 10px;">SibSp</th><th style="padding: 4px 10px;">Parch</th><th style="padding: 4px 10px;">Ticket</th><th style="padding: 4px 10px;">Fare</th><th style="padding: 4px 10px;">Cabin</th><th style="padding: 4px 10px;">Embarked</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="background:#fff8ee;">
                                <td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">Braund, Mr. Owen Harris</td><td style="padding: 4px 10px;">male</td><td style="padding: 4px 10px;">22</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">A/5 21171</td><td style="padding: 4px 10px;">7.2500</td><td style="padding: 4px 10px;">NaN</td><td style="padding: 4px 10px;">S</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td><td style="padding: 4px 10px;">female</td><td style="padding: 4px 10px;">38</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">PC 17599</td><td style="padding: 4px 10px;">71.2833</td><td style="padding: 4px 10px;">C85</td><td style="padding: 4px 10px;">C</td>
                            </tr>
                            <tr style="background:#fff8ee;">
                                <td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">Heikkinen, Miss. Laina</td><td style="padding: 4px 10px;">female</td><td style="padding: 4px 10px;">26</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">STON/O2. 3101282</td><td style="padding: 4px 10px;">7.9250</td><td style="padding: 4px 10px;">NaN</td><td style="padding: 4px 10px;">S</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">4</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">Futrelle, Mrs. Jacques Heath (Lily May Peel)</td><td style="padding: 4px 10px;">female</td><td style="padding: 4px 10px;">35</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">113803</td><td style="padding: 4px 10px;">53.1000</td><td style="padding: 4px 10px;">C123</td><td style="padding: 4px 10px;">S</td>
                            </tr>
                            <tr style="background:#fff8ee;">
                                <td style="padding: 4px 10px;">5</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">Allen, Mr. William Henry</td><td style="padding: 4px 10px;">male</td><td style="padding: 4px 10px;">35</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">373450</td><td style="padding: 4px 10px;">8.0500</td><td style="padding: 4px 10px;">NaN</td><td style="padding: 4px 10px;">S</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">6</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">Moran, Mr. James</td><td style="padding: 4px 10px;">male</td><td style="padding: 4px 10px;">&lt;NA&gt;</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">330877</td><td style="padding: 4px 10px;">8.4583</td><td style="padding: 4px 10px;">NaN</td><td style="padding: 4px 10px;">Q</td>
                            </tr>
                            <tr style="background:#fff8ee;">
                                <td style="padding: 4px 10px;">7</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">McCarthy, Mr. Timothy J</td><td style="padding: 4px 10px;">male</td><td style="padding: 4px 10px;">54</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">17463</td><td style="padding: 4px 10px;">51.8625</td><td style="padding: 4px 10px;">E46</td><td style="padding: 4px 10px;">S</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">8</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">Palsson, Master. Gosta Leonard</td><td style="padding: 4px 10px;">male</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">349909</td><td style="padding: 4px 10px;">21.0750</td><td style="padding: 4px 10px;">NaN</td><td style="padding: 4px 10px;">S</td>
                            </tr>
                            <tr style="background:#fff8ee;">
                                <td style="padding: 4px 10px;">9</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td><td style="padding: 4px 10px;">female</td><td style="padding: 4px 10px;">27</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">347742</td><td style="padding: 4px 10px;">11.1333</td><td style="padding: 4px 10px;">NaN</td><td style="padding: 4px 10px;">S</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">10</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">Nasser, Mrs. Nicholas (Adele Achem)</td><td style="padding: 4px 10px;">female</td><td style="padding: 4px 10px;">14</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">237736</td><td style="padding: 4px 10px;">30.0708</td><td style="padding: 4px 10px;">NaN</td><td style="padding: 4px 10px;">C</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div style="background:#d8f3dc; padding:15px; border-radius:4px; width:55%; max-width: 55%; margin-right: 20px;">
                <h3 style="text-align:center; margin-top: 0; margin-bottom: 10px; font-size: 16px;">Processed Dataset</h3>
                <div style="overflow-x:auto;">
                    <table style="width: 100%; border-collapse: collapse; text-align: right; background: white;">
                        <thead>
                            <tr style="border-bottom: 1px solid #aaa;">
                                <th style="padding: 4px 10px;">PassengerId</th><th style="padding: 4px 10px;">Survived</th><th style="padding: 4px 10px;">Pclass</th><th style="padding: 4px 10px;">Name</th><th style="padding: 4px 10px;">Sex</th><th style="padding: 4px 10px;">Age</th><th style="padding: 4px 10px;">SibSp</th><th style="padding: 4px 10px;">Parch</th><th style="padding: 4px 10px;">Ticket</th><th style="padding: 4px 10px;">Fare</th><th style="padding: 4px 10px;">Embarked_C</th><th style="padding: 4px 10px;">Embarked_Q</th><th style="padding: 4px 10px;">Embarked_S</th><th style="padding: 4px 10px;">Cabin_A</th><th style="padding: 4px 10px;">Cabin_B</th><th style="padding: 4px 10px;">Cabin_C</th><th style="padding: 4px 10px;">Cabin_D</th><th style="padding: 4px 10px;">Cabin_E</th><th style="padding: 4px 10px;">Cabin_F</th><th style="padding: 4px 10px;">Cabin_G</th><th style="padding: 4px 10px;">Cabin_T</th><th style="padding: 4px 10px;">Scaled_Age</th><th style="padding: 4px 10px;">Scaled_Fare</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="background:#f1faf3;">
                                <td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">Braund, Mr. Owen Harris</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">22</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">A/5 21171</td><td style="padding: 4px 10px;">7.2500</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">0.271174</td><td style="padding: 4px 10px;">0.014151</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">38</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">PC 17599</td><td style="padding: 4px 10px;">71.2833</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">0.472229</td><td style="padding: 4px 10px;">0.139136</td>
                            </tr>
                            <tr style="background:#f1faf3;">
                                <td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">Heikkinen, Miss. Laina</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">26</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">STON/O2. 3101282</td><td style="padding: 4px 10px;">7.9250</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">0.321438</td><td style="padding: 4px 10px;">0.015469</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">4</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">Futrelle, Mrs. Jacques Heath (Lily May Peel)</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">35</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">113803</td><td style="padding: 4px 10px;">53.1000</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">0.434531</td><td style="padding: 4px 10px;">0.103644</td>
                            </tr>
                            <tr style="background:#f1faf3;">
                                <td style="padding: 4px 10px;">5</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">Allen, Mr. William Henry</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">35</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">373450</td><td style="padding: 4px 10px;">8.0500</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">0.434531</td><td style="padding: 4px 10px;">0.015713</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">6</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">Moran, Mr. James</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">&lt;NA&gt;</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">330877</td><td style="padding: 4px 10px;">8.4583</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">NaN</td><td style="padding: 4px 10px;">0.016510</td>
                            </tr>
                            <tr style="background:#f1faf3;">
                                <td style="padding: 4px 10px;">7</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">McCarthy, Mr. Timothy J</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">54</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">17463</td><td style="padding: 4px 10px;">51.8625</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">0.673285</td><td style="padding: 4px 10px;">0.101229</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">8</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">Palsson, Master. Gosta Leonard</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">3</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">349909</td><td style="padding: 4px 10px;">21.0750</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">0.019854</td><td style="padding: 4px 10px;">0.041136</td>
                            </tr>
                            <tr style="background:#f1faf3;">
                                <td style="padding: 4px 10px;">9</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">27</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">2</td><td style="padding: 4px 10px;">347742</td><td style="padding: 4px 10px;">11.1333</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">0.334004</td><td style="padding: 4px 10px;">0.021731</td>
                            </tr>
                            <tr>
                                <td style="padding: 4px 10px;">10</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">Nasser, Mrs. Nicholas (Adele Achem)</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">14</td><td style="padding: 4px 10px;">1</td><td style="padding: 4px 10px;">0</td><td style="padding: 4px 10px;">237736</td><td style="padding: 4px 10px;">30.0708</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">True</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">False</td><td style="padding: 4px 10px;">0.170646</td><td style="padding: 4px 10px;">0.058694</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>`

      }

    ]
  }
];

// State Management
let hasCompletedOnce = sessionStorage.getItem('dp_completed') === 'true';

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

// Check if all steps are completed
function checkAllStepsCompleted() {
  return STATE.stepsStatus.every(status => status.completed);
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
  downloadBtn.id = 'downloadExperimentBtn';
  downloadBtn.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align: middle;">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="7 10 12 15 17 10"></polyline>
      <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
    Download Experiment
  `;
  downloadBtn.style.textAlign = 'center';
  downloadBtn.style.marginTop = "10px";

  // Check if all steps are completed (or were completed before a restart)
  const allCompleted = checkAllStepsCompleted() || hasCompletedOnce;

  if (allCompleted) {
    downloadBtn.style.backgroundColor = "#F57C2A"; // Orange when enabled
    downloadBtn.style.opacity = "1";
    downloadBtn.style.cursor = "pointer";
    downloadBtn.style.color = "white";
    downloadBtn.disabled = false;
    downloadBtn.onclick = downloadPDF;
  } else {
    downloadBtn.style.backgroundColor = "#f5f5f5"; // Light grey when disabled
    downloadBtn.style.opacity = "1";
    downloadBtn.style.cursor = "default";
    downloadBtn.style.color = "#9e9e9e";
    downloadBtn.style.border = "1px solid #e0e0e0";
    downloadBtn.disabled = false;
    downloadBtn.title = "Need to run the Experiment to download the pdf.";
    downloadBtn.onclick = function () {
      alert("Need to run the Experiment to download the pdf.");
    };
  }

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

    // Clear out any previous inline styles that wrapped the text
    codeHeaderBar.style.width = '';
    codeHeaderBar.style.maxWidth = '';
    codeHeaderBar.style.whiteSpace = '';
    codeHeaderBar.style.lineHeight = '';
    codeHeaderBar.style.textAlign = '';
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

  // Custom Zoom Button Logic for Interactive Imputation Cell
  const codeBoxContainer = document.querySelector('.code-box-container');
  let zoomBtn = document.getElementById('zoomCellBtn');
  if (!zoomBtn) {
    zoomBtn = document.createElement('button');
    zoomBtn.id = 'zoomCellBtn';
    zoomBtn.title = 'Zoom Code';
    zoomBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>';
    zoomBtn.style.position = "absolute";
    zoomBtn.style.right = "20px";
    zoomBtn.style.width = "40px";
    zoomBtn.style.height = "40px";
    zoomBtn.style.backgroundColor = "#F57C2A"; // Matches run button orange
    zoomBtn.style.color = "white";
    zoomBtn.style.border = "none";
    zoomBtn.style.borderRadius = "50%";
    zoomBtn.style.cursor = "pointer";
    zoomBtn.style.zIndex = "100";
    zoomBtn.style.display = "flex";
    zoomBtn.style.alignItems = "center";
    zoomBtn.style.justifyContent = "center";
    // Ensure container handles absolute children
    codeBoxContainer.style.position = "relative";
    codeBoxContainer.appendChild(zoomBtn);
  }

  const zoomInIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>';
  const zoomOutIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 14 10 14 10 20"></polyline><polyline points="20 10 14 10 14 4"></polyline><line x1="14" y1="10" x2="21" y2="3"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>';

  // Helper to place zoom button exactly underneath run button dynamically
  function updateZoomBtnPosition() {
    requestAnimationFrame(() => {
      const cRect = codeBoxContainer.getBoundingClientRect();
      const rRect = runBtn.getBoundingClientRect();
      const topOffset = (rRect.bottom - cRect.top) + 15;
      zoomBtn.style.top = topOffset + "px";
    });
  }

  if ((step.id === 'missing_values_imputation' && (STATE.subStepIndex === 1 || STATE.subStepIndex === 2)) ||
    (step.id === 'encoding_categorical_variables' && STATE.subStepIndex === 0) ||
    (step.id === 'normalization' && (STATE.subStepIndex === 0 || STATE.subStepIndex === 1)) ||
    (step.id === 'outlier_detection' && STATE.subStepIndex === 0) ||
    (step.id === 'data_comparison' && STATE.subStepIndex === 0)) {
    zoomBtn.style.display = "flex";
    updateZoomBtnPosition(); // Place below run button

    zoomBtn.onclick = function (e) {
      if (e) e.stopPropagation();
      if (codeBoxContainer.classList.contains('zoomed-in')) {
        codeBoxContainer.classList.remove('zoomed-in');
        codeBoxContainer.style.position = "relative";
        codeBoxContainer.style.top = "";
        codeBoxContainer.style.left = "";
        codeBoxContainer.style.width = "";
        codeBoxContainer.style.height = "";
        codeBoxContainer.style.zIndex = "";
        codeBoxContainer.style.backgroundColor = "";
        codeBoxContainer.style.borderRadius = "";
        codeBoxContainer.style.boxShadow = "";
        const codeBox = codeBoxContainer.querySelector('.code-box');
        if (codeBox) codeBox.style.paddingBottom = "";
        codeDisplay.style.color = "";
        zoomBtn.innerHTML = zoomInIcon;
        updateZoomBtnPosition(); // Reposition securely below runBtn
        zoomBtn.style.bottom = "auto";
        document.removeEventListener('click', window._codeZoomOutsideClick);
      } else {
        codeBoxContainer.classList.add('zoomed-in');

        // Exact positioning within the content area
        const contentArea = document.querySelector('.content-area');
        const rect = contentArea.getBoundingClientRect();

        codeBoxContainer.style.position = "fixed";
        codeBoxContainer.style.top = (rect.top + 10) + "px";
        codeBoxContainer.style.left = (rect.left + 10) + "px";
        codeBoxContainer.style.width = (rect.width - 20) + "px";
        codeBoxContainer.style.height = (rect.height - 20) + "px";
        const codeBox = codeBoxContainer.querySelector('.code-box');
        if (codeBox) codeBox.style.paddingBottom = "100px";
        codeBoxContainer.style.zIndex = "9999";
        codeBoxContainer.style.backgroundColor = "#2b2b2b";
        codeBoxContainer.style.borderRadius = "8px";
        codeBoxContainer.style.boxShadow = "0 10px 40px rgba(0,0,0,0.4)";
        codeDisplay.style.color = "#d4d4d4";
        zoomBtn.innerHTML = zoomOutIcon;
        // Keep it securely placed right below the run button at all times
        updateZoomBtnPosition();
        zoomBtn.style.bottom = "auto";

        window._codeZoomOutsideClick = function (evt) {
          if (codeBoxContainer.classList.contains('zoomed-in') && !codeBoxContainer.contains(evt.target) && !zoomBtn.contains(evt.target)) {
            zoomBtn.click();
          }
        };
        setTimeout(() => document.addEventListener('click', window._codeZoomOutsideClick), 50);
      }
    };

    // Reset zoom state on load
    codeBoxContainer.classList.remove('zoomed-in');
    codeBoxContainer.style.position = "relative";
    codeBoxContainer.style.top = "";
    codeBoxContainer.style.left = "";
    codeBoxContainer.style.width = "";
    codeBoxContainer.style.height = "";
    codeBoxContainer.style.zIndex = "";
    codeBoxContainer.style.backgroundColor = "";
    codeBoxContainer.style.borderRadius = "";
    codeBoxContainer.style.boxShadow = "";
    codeDisplay.style.color = "";
    zoomBtn.innerHTML = zoomInIcon;
    updateZoomBtnPosition();
    zoomBtn.style.bottom = "auto";
    if (window._codeZoomOutsideClick) document.removeEventListener('click', window._codeZoomOutsideClick);
  } else {
    zoomBtn.style.display = "none";
    codeBoxContainer.classList.remove('zoomed-in');
    codeBoxContainer.style.position = "relative";
    codeBoxContainer.style.top = "";
    codeBoxContainer.style.left = "";
    codeBoxContainer.style.width = "";
    codeBoxContainer.style.height = "";
    codeBoxContainer.style.zIndex = "";
    codeBoxContainer.style.backgroundColor = "";
    codeBoxContainer.style.borderRadius = "";
    codeBoxContainer.style.boxShadow = "";
    codeDisplay.style.color = "";
  }
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
    outputContent.innerHTML = typeof block.output === 'function' ? block.output() : block.output;
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

    // Check if this is the missing values imputation block
    if (document.getElementById('simMethodDropdown')) {
      window.imputationApplied = false;
      window.cabinImputationApplied = false;
      window.embarkedImputationApplied = false;
      document.getElementById('simMethodDropdown').value = 'Mean';
      document.getElementById('simConstantInput').value = '0';
      window.updateSimPreview && window.updateSimPreview();
    }

    // Reset encoding interactive state
    if (document.getElementById('encMethodDropdown')) {
      if (window.encodedFeatures) window.encodedFeatures.clear();
      document.getElementById('encMethodDropdown').value = "Label Encoding";
      window.onEncMethodChange && window.onEncMethodChange();
    }

    // Reset scaling interactive state
    if (document.getElementById('scaleMethodDropdown')) {
      if (window.scaledFeatures) window.scaledFeatures.clear();
      document.getElementById('scaleMethodDropdown').value = "StandardScaler";
      document.getElementById('scaleFeatureDropdown').value = "Age";
      window.onScaleFeatureChange && window.onScaleFeatureChange();
    }

    // Reset outlier interactive state
    if (document.getElementById('outlierMethodDropdown')) {
      document.getElementById('outlierMethodDropdown').value = "IQR";
      document.getElementById('outlierFeatureDropdown').value = "Age";
      window.onOutlierChange && window.onOutlierChange();
    }

    // Reset outlier interactive state
    if (document.getElementById('outlierMethodDropdown')) {
      document.getElementById('outlierMethodDropdown').value = "IQR";
      document.getElementById('outlierFeatureDropdown').value = "Age";
      window.onOutlierChange && window.onOutlierChange();
    }

    // 5. Handle Next Logic
    const hasNextBlock = STATE.subStepIndex < step.blocks.length - 1;

    const enableNextLogic = () => {
      if (hasNextBlock) {
        // Wait 1s then change button to "Next"
        setTimeout(() => {
          runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
          runBtn.style.backgroundColor = '#5FA8E4'; // Orange
          runBtn.disabled = false;

          // Switch handler to Next
          runBtn.onclick = function () {
            if (document.getElementById('simMethodDropdown') && !window.imputationApplied) {
              alert("Need to impute the missing value to go on next step");
              return;
            }
            if (document.getElementById('catSimPreviewArea') && (!window.cabinImputationApplied || !window.embarkedImputationApplied)) {
              alert("Need to impute the categorical missing values to go on next step");
              return;
            }
            if (document.getElementById('encMethodDropdown') && (!window.encodedFeatures || window.encodedFeatures.size < 4)) {
              alert("Need to encode all categorical features to go on next step");
              return;
            }
            if (document.getElementById('scaleMethodDropdown') && (!window.scaledFeatures || window.scaledFeatures.size < 2)) {
              alert("Need to scale all features to go on next step");
              return;
            }
            nextSubStep();
          };
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
              if (document.getElementById('simMethodDropdown') && !window.imputationApplied) {
                alert("Need to impute the missing value to go on next step");
                return;
              }
              if (document.getElementById('catSimPreviewArea') && (!window.cabinImputationApplied || !window.embarkedImputationApplied)) {
                alert("Need to impute the categorical missing values to go on next step");
                return;
              }
              if (document.getElementById('encMethodDropdown') && (!window.encodedFeatures || window.encodedFeatures.size < 4)) {
                alert("Need to encode all categorical features to go on next step");
                return;
              }
              if (document.getElementById('scaleMethodDropdown') && (!window.scaledFeatures || window.scaledFeatures.size < 2)) {
                alert("Need to scale all features to go on next step");
                return;
              }
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
            runBtn.onclick = function () {
              if (document.getElementById('simMethodDropdown') && !window.imputationApplied) {
                alert("Need to impute the missing value to go on next step");
                return;
              }
              if (document.getElementById('catSimPreviewArea') && (!window.cabinImputationApplied || !window.embarkedImputationApplied)) {
                alert("Need to impute the categorical missing values to go on next step");
                return;
              }
              if (document.getElementById('encMethodDropdown') && (!window.encodedFeatures || window.encodedFeatures.size < 4)) {
                alert("Need to encode all categorical features to go on next step");
                return;
              }
              if (document.getElementById('scaleMethodDropdown') && (!window.scaledFeatures || window.scaledFeatures.size < 2)) {
                alert("Need to scale all features to go on next step");
                return;
              }
              showCompletionMessage();
            };
          }, 500);
        }
      }
    };

    enableNextLogic();

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
    .replace(/(^#.*$)/gm, '<span style="color: green;">$1</span>')
    .replace(/(\s#.*$)/gm, '<span style="color: green;">$1</span>')
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
  container.innerHTML = `<img src="./images/${feature}.png" alt="Sigmoid of ${feature}" style="max-height:300px; border:none; display:block;">`;
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
  hasCompletedOnce = true;
  sessionStorage.setItem('dp_completed', 'true');
  outputContent.innerHTML = ''; // Clear output content
  bottomPane.classList.add('active-output');
  bottomPane.style.display = 'flex';
  bottomPane.style.flexDirection = 'column';
  bottomPane.style.justifyContent = 'center';
  bottomPane.style.alignItems = 'center';

  const msgHTML = `
    <style>
      @keyframes clap {
        0%, 100% { transform: rotate(-15deg) scale(1); }
        50% { transform: rotate(15deg) scale(1.1); }
      }
      .clapping-hands {
        display: inline-block;
        font-size: 2.5rem;
        animation: clap 0.5s ease-in-out infinite;
        margin: 0 5px;
      }
    </style>
    <div style="text-align: center; animation: fadeIn 1s ease;">
      <div style="margin-bottom: 20px;">
        <span class="clapping-hands">👏</span>
        <span class="clapping-hands" style="animation-delay: 0.15s;">👏</span>
        <span class="clapping-hands" style="animation-delay: 0.3s;">👏</span>
      </div>
      <h1 style="color: #2a9d8f; font-size: 2.5rem; margin-bottom: 20px;">Congratulations!</h1>
      <p style="font-size: 1.2rem; color: #333; max-width: 600px; margin: 0 auto;">You have completed your first Machine Learning experiment and learned how to clean, transform, and visualize data—an essential step for building effective machine learning models.</p>
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
  link.href = './Exp-1.pdf';
  link.download = 'Exp-1.pdf';
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

// Global scope for Imputation Interface
window.updateSimPreview = function () {
  const method = document.getElementById("simMethodDropdown").value;
  const constantDiv = document.getElementById("simConstantDiv");
  if (method === "Constant") {
    constantDiv.style.display = "block";
  } else {
    constantDiv.style.display = "none";
  }

  let label = method;
  let value = 0;
  if (method === "Mean") value = 30;
  if (method === "Median") value = 28;
  if (method === "Mode") value = 24;
  if (method === "Constant") value = parseInt(document.getElementById("simConstantInput").value) || 0;

  const previewArea = document.getElementById("simPreviewArea");
  if (!previewArea) return;

  const applied = window.imputationApplied || false;
  const missingCount = applied ? 0 : 177;

  let btnConfig = applied
    ? `background-color: #cccccc; color: #666666; cursor: not-allowed;`
    : `background-color: #62a35e; color: white; cursor: pointer;`;

  let contentHTML = `<div style="text-align: center; font-family: sans-serif; margin-bottom: 5px; width: 350px;">
      <div style="font-size: 13px;">The numeric feature 'Age' contains 177 missing values.</div>
      <div style="color: #1d3557; font-size: 14px; margin-bottom: 3px;">
          ${missingCount} NaN missing values will be replaced with ${label} ${value}
      </div>
      <div style="color: red; font-size: 20px; text-align: center;">
          &uarr;
      </div>
  </div>
  <div style="text-align: left; padding-left: 20px; margin-bottom: 15px;">
      <button id="simApplyBtn" onclick="window.applySimImputation()" ${applied ? 'disabled' : ''} style="padding: 10px 20px; ${btnConfig} border: none; border-radius: 2px; font-size: 13px; font-weight: bold; width: 330px;">Impute the Missing Values</button>
  </div>
  <div style="text-align: left; font-family: monospace; font-size: 13px; padding-left: 20px; margin-bottom: 5px;">
      Random 10 Sample Showing Imputation Effect<br><br>
     
  </div>`;

  let tableHTML = `<table style="width: auto; background-color: white; color: black; margin-left: 20px; text-align: right; font-family: monospace; font-size: 13px; border-collapse: collapse;">
  <thead><tr style="border-bottom: 1px solid #ccc; height: 30px;"><th style="padding: 0 15px;">Before_Imputation_Age</th><th style="padding: 0 15px;">After_Imputation_Age</th></tr></thead>
  <tbody>
    <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">&lt;NA&gt;</td><td style="padding: 0 15px;">${applied ? value : ""}</td></tr>
    <tr style="height: 25px;"><td style="padding: 0 15px;">&lt;NA&gt;</td><td style="padding: 0 15px;">${applied ? value : ""}</td></tr>
    <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">&lt;NA&gt;</td><td style="padding: 0 15px;">${applied ? value : ""}</td></tr>
    <tr style="height: 25px;"><td style="padding: 0 15px;">&lt;NA&gt;</td><td style="padding: 0 15px;">${applied ? value : ""}</td></tr>
    <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">&lt;NA&gt;</td><td style="padding: 0 15px;">${applied ? value : ""}</td></tr>
    <tr style="height: 25px;"><td style="padding: 0 15px;">35</td><td style="padding: 0 15px;">${applied ? 35 : ""}</td></tr>
    <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">42</td><td style="padding: 0 15px;">${applied ? 42 : ""}</td></tr>
    <tr style="height: 25px;"><td style="padding: 0 15px;">20</td><td style="padding: 0 15px;">${applied ? 20 : ""}</td></tr>
    <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">36</td><td style="padding: 0 15px;">${applied ? 36 : ""}</td></tr>
    <tr style="height: 25px;"><td style="padding: 0 15px;">40</td><td style="padding: 0 15px;">${applied ? 40 : ""}</td></tr>
  </tbody></table>`;

  previewArea.innerHTML = contentHTML + tableHTML;
};

window.applySimImputation = function () {
  window.imputationApplied = true;
  window.updateSimPreview();
  if (window.continueRunStep) {
    window.continueRunStep();
    window.continueRunStep = null;
  }
};

window.updateCatSimPreview = function () {
  const previewArea = document.getElementById("catSimPreviewArea");
  if (!previewArea) return;

  const cabinApplied = window.cabinImputationApplied || false;
  const embarkedApplied = window.embarkedImputationApplied || false;

  const cabinMissingCount = cabinApplied ? 0 : 687;
  const embarkedMissingCount = embarkedApplied ? 0 : 2;

  let contentHTML = `
  <div style="font-size: 13px; font-family: monospace; margin-bottom: 20px;">
    Missing values in Embarked: ${embarkedMissingCount}<br>
    Missing values in Cabin: ${cabinMissingCount}<br><br>
    Most frequent value in Embarked: S<br>
    Most frequent value in Cabin: B96 B98<br><br>
    Concept Explanation:<br><br>
    Categorical missing values are commonly filled using the<br>
    most frequent category (also called the mode).<br><br>
    Reason:<br>
    Replacing missing values with the most frequent category<br>
    preserves the distribution of the dataset and avoids introducing<br>
    new artificial categories.<br><br>
    In this dataset:<br>
    Embarked missing values will be replaced with the most frequent port.<br>
    Cabin missing values will be replaced with the most frequent cabin value.<br><br>
    Once the button is pressed, the missing values will be filled<br>
    and the dataset will be updated.
  </div>
  
  <div style="margin-bottom: 30px;">
    <button onclick="window.applyCabinImputation()" ${cabinApplied ? 'disabled' : ''} style="padding: 10px 20px; background-color: ${cabinApplied ? '#cccccc' : '#62a35e'}; color: ${cabinApplied ? '#666666' : 'white'}; border: none; border-radius: 2px; cursor: ${cabinApplied ? 'not-allowed' : 'pointer'}; font-size: 13px; font-weight: bold; width: 330px; margin-bottom: 20px;">Impute Missing Values for Cabin</button>
    <div style="font-family: monospace; font-size: 13px; margin-bottom: 15px;">
        6 Random Samples Showing Cabin Imputation Effect
    </div>
    <table style="width: auto; background-color: transparent; text-align: right; font-family: monospace; font-size: 13px; border-collapse: collapse; margin-left: 20px;">
      <thead><tr style="border-bottom: 1px solid #ccc; height: 30px;"><th style="padding: 0 15px;">Before_Cabin</th><th style="padding: 0 15px;">After_Cabin</th></tr></thead>
      <tbody>
        <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">NaN</td><td style="padding: 0 15px;">${cabinApplied ? 'B96 B98' : ''}</td></tr>
        <tr style="height: 25px;"><td style="padding: 0 15px;">NaN</td><td style="padding: 0 15px;">${cabinApplied ? 'B96 B98' : ''}</td></tr>
        <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">E36</td><td style="padding: 0 15px;">${cabinApplied ? 'E36' : ''}</td></tr>
        <tr style="height: 25px;"><td style="padding: 0 15px;">NaN</td><td style="padding: 0 15px;">${cabinApplied ? 'B96 B98' : ''}</td></tr>
        <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">F33</td><td style="padding: 0 15px;">${cabinApplied ? 'F33' : ''}</td></tr>
        <tr style="height: 25px;"><td style="padding: 0 15px;">NaN</td><td style="padding: 0 15px;">${cabinApplied ? 'B96 B98' : ''}</td></tr>
      </tbody>
    </table>
  </div>
  
  <div>
    <button onclick="window.applyEmbarkedImputation()" ${embarkedApplied ? 'disabled' : ''} style="padding: 10px 20px; background-color: ${embarkedApplied ? '#cccccc' : '#62a35e'}; color: ${embarkedApplied ? '#666666' : 'white'}; border: none; border-radius: 2px; cursor: ${embarkedApplied ? 'not-allowed' : 'pointer'}; font-size: 13px; font-weight: bold; width: 330px; margin-bottom: 20px;">Impute Missing Values for Embarked</button>
    <div style="font-family: monospace; font-size: 13px; margin-bottom: 15px;">
        6 Random Samples Showing Embarked Imputation Effect
    </div>
    <table style="width: auto; background-color: transparent; text-align: right; font-family: monospace; font-size: 13px; border-collapse: collapse; margin-left: 20px;">
      <thead><tr style="border-bottom: 1px solid #ccc; height: 30px;"><th style="padding: 0 15px;">Before_Embarked</th><th style="padding: 0 15px;">After_Embarked</th></tr></thead>
      <tbody>
        <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">NaN</td><td style="padding: 0 15px;">${embarkedApplied ? 'S' : ''}</td></tr>
        <tr style="height: 25px;"><td style="padding: 0 15px;">NaN</td><td style="padding: 0 15px;">${embarkedApplied ? 'S' : ''}</td></tr>
        <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">S</td><td style="padding: 0 15px;">${embarkedApplied ? 'S' : ''}</td></tr>
        <tr style="height: 25px;"><td style="padding: 0 15px;">C</td><td style="padding: 0 15px;">${embarkedApplied ? 'C' : ''}</td></tr>
        <tr style="background:#f2f2f2; height: 25px;"><td style="padding: 0 15px;">Q</td><td style="padding: 0 15px;">${embarkedApplied ? 'Q' : ''}</td></tr>
        <tr style="height: 25px;"><td style="padding: 0 15px;">C</td><td style="padding: 0 15px;">${embarkedApplied ? 'C' : ''}</td></tr>
      </tbody>
    </table>
  </div>
  `;

  previewArea.innerHTML = contentHTML;
};

window.applyCabinImputation = function () {
  window.cabinImputationApplied = true;
  window.updateCatSimPreview();
};

window.applyEmbarkedImputation = function () {
  window.embarkedImputationApplied = true;
  window.updateCatSimPreview();
};

window.onEncMethodChange = function () {
  const methodSelect = document.getElementById("encMethodDropdown");
  const featureSelect = document.getElementById("encFeatureDropdown");
  if (!methodSelect || !featureSelect) return;

  const method = methodSelect.value;
  window.encodingMethod = method;

  const available = window.featureMap[method].filter(f => !window.encodedFeatures.has(f));
  featureSelect.innerHTML = "";
  if (available.length === 0) {
    featureSelect.disabled = true;
  } else {
    featureSelect.disabled = false;
    available.forEach(f => {
      let opt = document.createElement("option");
      opt.value = f;
      opt.text = f;
      featureSelect.appendChild(opt);
    });
    window.encodingFeature = available[0];
  }
  window.updateEncodingPreview();
};

window.applyEncoding = function () {
  const featureSelect = document.getElementById("encFeatureDropdown");
  if (!featureSelect || featureSelect.disabled) return;
  const feature = featureSelect.value;
  window.encodedFeatures.add(feature);
  window.updateEncodingPreview();
};

window.updateEncodingPreview = function () {
  const previewArea = document.getElementById("encodingPreviewArea");
  const methodSelect = document.getElementById("encMethodDropdown");
  const featureSelect = document.getElementById("encFeatureDropdown");
  if (!previewArea || !methodSelect || !featureSelect) return;

  if (!featureSelect.disabled) window.encodingFeature = featureSelect.value;
  const method = methodSelect.value;
  const feature = featureSelect.disabled ? null : featureSelect.value;

  let textHTML = "";
  if (feature) {
    textHTML += `<div style="font-family: monospace; font-size: 13px; margin-bottom: 15px;">
            Selected Encoding Method: ${method}<br>
            Selected Feature: ${feature}<br>
        </div>`;
  }

  if (method === "Label Encoding") {
    textHTML += `
        <div style="font-family: monospace; font-size: 13px; margin-bottom: 20px;">
            Label Encoding:<br>
            Label Encoding converts categorical values into numbers.<br><br>
            For the feature 'Sex', the two categories are male and female.<br><br>
            Mapping used in this dataset:<br>
            female &rarr; 0<br>
            male &rarr; 1<br><br>
            The numbers 0 and 1 act only as identifiers so machine learning<br>
            models can process the categorical feature.
        </div>`;
  } else if (method === "One-Hot Encoding") {
    textHTML += `
        <div style="font-family: monospace; font-size: 13px; margin-bottom: 20px;">
            One-Hot Encoding:<br>
            One-Hot Encoding creates separate binary columns for each category.<br><br>`;

    if (feature === "Embarked") {
      textHTML += `For 'Embarked', passengers boarded from four ports:<br>
            C (Cherbourg), Q (Queenstown), S (Southampton), and E (Eastport).<br><br>
            Therefore four columns are created:<br>
            Embarked_C, Embarked_Q, Embarked_S, Embarked_E.<br><br>
            Each row contains 1 in the column representing the category<br>
            and 0 in the remaining columns.`;
    } else if (feature === "Cabin" || !feature) {
      textHTML += `For 'Cabin', the deck letter is extracted first.<br>
            Possible decks include A, B, C, D, E, F, G and T.<br><br>
            One column is created for each deck such as Cabin_A, Cabin_B etc.<br>
            A value of 1 indicates the passenger belongs to that deck.`;
    }
    textHTML += `</div>`;
  } else if (method === "Ordinal Encoding") {
    textHTML += `
        <div style="font-family: monospace; font-size: 13px; margin-bottom: 20px;">
            Ordinal Encoding:<br>
            Ordinal Encoding is used when categories have a natural ranking.<br><br>
            In the Titanic dataset, 'Pclass' represents passenger class:<br>
            1 = First Class, 2 = Second Class, 3 = Third Class.<br><br>
            The encoder assigns increasing integers:<br><br>
            1 &rarr; 0<br>
            2 &rarr; 1<br>
            3 &rarr; 2
        </div>`;
  }

  let beforeData = [];
  let afterCols = [];
  let afterData = [];
  let isEncoded = feature && window.encodedFeatures.has(feature);

  if (feature === "Sex") {
    beforeData = ["male", "female", "female", "female", "male"];
    afterCols = ["Sex"];
    afterData = isEncoded ? [[1], [0], [0], [0], [1]] : [[""], [""], [""], [""], [""]];
  } else if (feature === "Embarked") {
    beforeData = ["S", "C", "E", "S", "S"];
    afterCols = ["Embarked_C", "Embarked_Q", "Embarked_S", "Embarked_E"];
    afterData = isEncoded ?
      [["False", "False", "True", "False"], ["True", "False", "False", "False"], ["False", "False", "False", "True"], ["False", "False", "True", "False"], ["False", "False", "True", "False"]] :
      [Array(4).fill(""), Array(4).fill(""), Array(4).fill(""), Array(4).fill(""), Array(4).fill("")];
  } else if (feature === "Cabin") {
    beforeData = ["B", "C", "D", "C", "E"];
    afterCols = ["Cabin_A", "Cabin_B", "Cabin_C", "Cabin_D", "Cabin_E", "Cabin_F", "Cabin_G", "Cabin_T"];
    afterData = isEncoded ?
      [["False", "True", "False", "False", "False", "False", "False", "False"],
      ["False", "False", "True", "False", "False", "False", "False", "False"],
      ["False", "False", "False", "True", "False", "False", "False", "False"],
      ["False", "False", "True", "False", "False", "False", "False", "False"],
      ["False", "False", "False", "False", "True", "False", "False", "False"]] :
      [Array(8).fill(""), Array(8).fill(""), Array(8).fill(""), Array(8).fill(""), Array(8).fill("")];
  } else if (feature === "Pclass") {
    beforeData = ["3", "1", "3", "1", "3"];
    afterCols = ["Pclass"];
    afterData = isEncoded ? [[2], [0], [2], [0], [2]] : [[""], [""], [""], [""], [""]];
  }

  let btnText = feature ? `Encode ${feature}` : "No Feature Available";
  let btnDisabled = !feature || isEncoded ? "disabled" : "";
  let btnStyle = (!feature || isEncoded) ? "background-color: #f5f5f5; color: #aaa; cursor: default;" : "background-color: #62a35e; color: white; cursor: pointer;";

  if (!feature) {
    previewArea.innerHTML = `<div style="font-family: monospace; font-size: 13px;">` + textHTML + `</div>` +
      `<button disabled style="padding: 10px 20px; border: none; border-radius: 2px; font-size: 13px; font-weight: bold; width: 220px; margin-top: 10px; margin-bottom: 20px; ${btnStyle}">${btnText}</button>`;
    return;
  }

  let tablesHTML = `
    <div style="display: flex; gap: 40px; margin-bottom: 20px; font-family: sans-serif;">
        <div style="width: auto;">
            <div style="font-weight: bold; font-size: 14px; margin-bottom: 10px;">Dataset Before Encoding</div>
            <table style="text-align: center; border-collapse: collapse; font-family: monospace; font-size: 13px; width: 100%;">
                <thead>
                    <tr style="border-bottom: 1px solid #ccc; height: 30px;">
                        <th style="padding: 0 15px;">${feature}</th>
                    </tr>
                </thead>
                <tbody>`;

  for (let i = 0; i < beforeData.length; i++) {
    let bg = i % 2 === 0 ? "background:#f2f2f2;" : "";
    tablesHTML += `<tr style="${bg} height: 25px;"><td style="padding: 0 15px;">${beforeData[i]}</td></tr>`;
  }

  tablesHTML += `</tbody></table>
        </div>
        <div style="width: auto;">
            <div style="font-weight: bold; font-size: 14px; margin-bottom: 10px;">Dataset After Encoding</div>
            <table style="text-align: center; border-collapse: collapse; font-family: monospace; font-size: 13px; width: 100%;">
                <thead>
                    <tr style="border-bottom: 1px solid #ccc; height: 30px;">`;

  afterCols.forEach(col => {
    tablesHTML += `<th style="padding: 0 15px;">${col}</th>`;
  });

  tablesHTML += `</tr></thead><tbody>`;
  for (let i = 0; i < afterData.length; i++) {
    let bg = i % 2 === 0 ? "background:#f2f2f2;" : "";
    tablesHTML += `<tr style="${bg} height: 25px;">`
    afterData[i].forEach(val => {
      tablesHTML += `<td style="padding: 0 15px;">${val}</td>`;
    });
    tablesHTML += `</tr>`;
  }

  tablesHTML += `</tbody></table>
        </div>
    </div>`;

  previewArea.innerHTML = `<div style="font-family: monospace; font-size: 13px;">` + textHTML + `</div>` + tablesHTML +
    `<button ${btnDisabled} onclick="window.applyEncoding()" style="padding: 10px 20px; border: none; border-radius: 2px; font-size: 13px; font-weight: bold; width: 220px; margin-top: 10px; ${btnStyle}">${btnText}</button>`;
};

window.onScaleFeatureChange = function () {
  const fSelect = document.getElementById("scaleFeatureDropdown");
  const feature = fSelect ? fSelect.value : "Age";
  window.scaleFeature = feature;
  window.updateScalingPreview();
};

window.onScaleMethodChange = function () {
  const mSelect = document.getElementById("scaleMethodDropdown");
  const method = mSelect ? mSelect.value : "StandardScaler";
  window.scaleMethod = method;
  window.updateScalingPreview();
};

window.applyScaling = function () {
  const feature = window.scaleFeature;
  const method = window.scaleMethod;
  if (!window.scaledMethods) window.scaledMethods = {};
  window.scaledMethods[feature] = method;
  window.scaledFeatures.add(feature);
  window.updateScalingPreview();
};

window.updateScalingPreview = function () {
  const previewArea = document.getElementById("scalingPreviewArea");
  if (!previewArea) return;

  let btnText = `Scale ${window.scaleFeature}`;
  let isScaled = window.scaledFeatures.has(window.scaleFeature);
  let btnDisabled = isScaled ? "disabled" : "";
  let btnStyle = isScaled ? "background-color: #BDBDBD; color: #555; cursor: default;" : "background-color: #62a35e; color: white; cursor: pointer;";
  if (isScaled) btnText = `${window.scaleFeature} Already Scaled`;

  let explanationHTML = `
        <div style="font-family: monospace; font-size: 13px; margin-bottom: 20px;">
            Concept Explanation:<br>
            Feature scaling transforms numerical values so that different features operate on comparable ranges. This prevents machine learning algorithms from being biased toward features with larger magnitudes.<br><br>
    `;

  if (window.scaleMethod === "StandardScaler") {
    explanationHTML += `StandardScaler (Standardization)<br><br>
            This method transforms the feature so that the resulting values<br>
            have a mean of 0 and a standard deviation of 1.<br><br>
            Formula:<br>
            z = (x &minus; &mu;) / &sigma;<br><br>
            Where:<br>
            x = original value<br>
            &mu; = mean<br>
            &sigma; = standard deviation`;
  } else if (window.scaleMethod === "MinMaxScaler") {
    explanationHTML += `MinMaxScaler (Normalization)<br><br>
            This method rescales the data to a fixed range, usually between 0 and 1.<br><br>
            Formula:<br>
            x' = (x &minus; xmin) / (xmax &minus; xmin)`;
  } else {
    explanationHTML += `RobustScaler<br><br>
            This method scales the data using the median and the interquartile range.<br><br>
            Formula:<br>
            x' = (x &minus; median) / IQR`;
  }
  explanationHTML += `</div>`;

  let visualizationsHTML = "";
  let methodUsed = window.scaledMethods ? window.scaledMethods[window.scaleFeature] : null;

  if (isScaled && methodUsed === window.scaleMethod) {
    let actualData = [];
    let scaledData = [];
    let img2 = "";

    if (window.scaleFeature === "Age") {
      if (window.scaleMethod === "StandardScaler") {
        actualData = [29, 34, 36, 20, 24, 70, 33, 19, 34, 58, 35, 24, 23, 36, 38, 34, 22, 42, 42, 27, 28, 0, 24, 20, 26, 15, 17, 33, 48, 49];
        scaledData = ["-0.048", "0.296", "0.434", "-0.668", "-0.393", "2.776", "0.227", "-0.737", "0.296", "1.950", "0.365", "-0.393", "-0.461", "0.434", "0.572", "0.296", "-0.530", "0.847", "0.847", "-0.186", "-0.117", "-2.017", "-0.393", "-0.668", "-0.255", "-1.013", "-0.875", "0.227", "1.261", "1.330"];
        img2 = "age_standardscaler.png";
      } else if (window.scaleMethod === "MinMaxScaler") {
        actualData = [27, 36, 23, 33, 40, 30, 6, 22, 36, 62, 16, 25, 20, 30, 37, 18, 29, 2, 38, 16, 64, 20, 39, 25, 8, 30, 42, 33, 19, 4];
        scaledData = ["0.334", "0.447", "0.284", "0.409", "0.497", "0.372", "0.070", "0.271", "0.453", "0.774", "0.196", "0.309", "0.246", "0.372", "0.460", "0.221", "0.359", "0.020", "0.472", "0.196", "0.799", "0.246", "0.485", "0.309", "0.095", "0.372", "0.522", "0.409", "0.233", "0.045"];
        img2 = "age_minmaxscaler.png";
      } else {
        actualData = [42, 49, 2, 29, 43, 56, 27, 23, 45, 8, 18, 22, 25, 33, 2, 19, 19, 24, 49, 22, 2, 24, 49, 55, 41, 34, 64, 30, 3, 20];
        scaledData = ["0.783", "1.175", "-1.455", "0.056", "0.839", "1.566", "-0.056", "-0.280", "0.951", "-1.119", "-0.559", "-0.336", "-0.168", "0.280", "-1.455", "-0.503", "-0.503", "-0.224", "1.175", "-0.336", "-1.455", "-0.224", "1.175", "1.538", "0.727", "0.336", "2.014", "0.112", "-1.399", "-0.448"];
        img2 = "age_robustscaler.png";
      }
    } else {
      if (window.scaleMethod === "StandardScaler") {
        actualData = [8, 11, 17, 7, 6, 0, 26, 31, 13, 9, 6, 26, 13, 13, 10, 14, 7, 26, 7, 24, 14, 7, 24, 7, 56, 11, 11, 83, 7, 7];
        scaledData = ["-0.474", "-0.422", "-0.290", "-0.494", "-0.508", "-0.648", "-0.125", "-0.016", "-0.387", "-0.457", "-0.518", "-0.125", "-0.387", "-0.377", "-0.437", "-0.357", "-0.503", "-0.120", "-0.490", "-0.162", "-0.358", "-0.489", "-0.165", "-0.501", "0.489", "-0.424", "-0.417", "1.026", "-0.492", "-0.489"];
        img2 = "fare_standardscaler.png";
      } else if (window.scaleMethod === "MinMaxScaler") {
        actualData = [93, 13, 14, 8, 8, 7, 26, 7, 7, 146, 0, 63, 7, 30, 7, 7, 7, 83, 52, 14, 7, 23, 11, 32, 7, 39, 7, 7, 7, 263];
        scaledData = ["0.182", "0.026", "0.028", "0.016", "0.016", "0.014", "0.051", "0.015", "0.014", "0.286", "0.000", "0.124", "0.015", "0.059", "0.015", "0.014", "0.015", "0.163", "0.103", "0.028", "0.015", "0.045", "0.022", "0.063", "0.015", "0.077", "0.015", "0.014", "0.014", "0.513"];
        img2 = "fare_minmaxscaler.png";
      } else {
        actualData = [7, 34, 10, 9, 7, 108, 47, 512, 5, 8, 14, 55, 27, 31, 31, 30, 13, 7, 7, 15, 18, 7, 14, 30, 7, 14, 8, 27, 26, 7];
        scaledData = ["-0.313", "0.863", "-0.171", "-0.216", "-0.289", "4.090", "1.414", "21.563", "-0.409", "-0.277", "0.000", "1.775", "0.575", "0.717", "0.729", "0.673", "-0.063", "-0.317", "-0.290", "0.056", "0.186", "-0.284", "0.000", "0.673", "-0.313", "0.002", "-0.277", "0.576", "0.524", "-0.288"];
        img2 = "fare_robustscaler.png";
      }
    }

    let visuals = `<div style="text-align: left; margin-bottom: 20px;">
                <img src="images/${img2}" onerror="this.src=''; this.alt='Scaling Visualization'" style="width: 700px; border: 1px solid #aaa;" />
        </div>`;

    let tableRows = "";
    for (let i = 0; i < actualData.length; i++) {
      let bg = i % 2 === 0 ? "background:#f2f2f2;" : "";
      tableRows += `<tr style="${bg} height: 25px;">
                <td style="padding: 0 15px;">${actualData[i]}</td>
                <td style="padding: 0 15px;">${scaledData[i]}</td>
            </tr>`;
    }

    let tableHTML = `<div style="font-family: monospace; font-size: 13px; font-weight: bold; margin-bottom: 5px; margin-top: 10px;">Sample Values Showing Scaling Effect</div>
        <div style="height:220px; overflow-y:auto; border:1px solid #ccc; width: fit-content; margin-bottom: 20px;">
            <table style="text-align: right; border-collapse: collapse; font-family: monospace; font-size: 13px; width: 100%;">
                <thead style="position: sticky; top: 0; background: white;">
                    <tr style="border-bottom: 1px solid #ccc; height: 30px;">
                        <th style="padding: 0 15px;">Actual_${window.scaleFeature}</th>
                        <th style="padding: 0 15px;">Scaled_${window.scaleFeature}</th>
                    </tr>
                </thead>
                <tbody>
                    ${tableRows}
                </tbody>
            </table>
        </div>
        <div style="font-family: monospace; font-size: 13px;">Scaled ${window.scaleFeature} values saved in df_clean as column: Scaled_${window.scaleFeature}</div>`;

    visualizationsHTML = visuals + tableHTML;
  }

  previewArea.innerHTML = `<div style="margin-left: -20px; margin-bottom: 20px;"><button ${btnDisabled} onclick="window.applyScaling()" style="padding: 10px 20px; border: none; border-radius: 2px; font-size: 13px; font-weight: bold; width: auto; min-width: 150px; ${btnStyle}">${btnText}</button></div>` +
    explanationHTML + visualizationsHTML;
};

window.onOutlierChange = function () {
  window.updateOutlierPreview();
};

window.updateOutlierPreview = function () {
  const previewArea = document.getElementById("outlierPreviewArea");
  const mSelect = document.getElementById("outlierMethodDropdown");
  const fSelect = document.getElementById("outlierFeatureDropdown");
  if (!previewArea || !mSelect || !fSelect) return;

  let method = mSelect.value;
  let feature = fSelect.value;

  let dataPoints = feature === "Age" ? 714 : 891;
  let outCount = 0;
  if (feature === "Age" && method === "IQR") outCount = 1;
  else if (feature === "Age" && method === "Z-Score") outCount = 2;
  else if (feature === "Fare" && method === "IQR") outCount = 34;
  else if (feature === "Fare" && method === "Z-Score") outCount = 20;

  let explanationHTML = "";
  if (method === "IQR") {
    let multiplier = feature === "Fare" ? 4.5 : 2.3;
    explanationHTML += `
IQR (Interquartile Range) Method<br><br>
The IQR method detects outliers by examining the spread<br>
of the middle 50% of the dataset.<br><br>
Formula:<br>
IQR = Q3 &minus; Q1<br><br>
Where:<br>
Q1 = 25th percentile<br>
Q3 = 75th percentile<br><br>
Outlier limits:<br><br>
Lower Bound = Q1 &minus; ${multiplier} &times; IQR<br>
Upper Bound = Q3 + ${multiplier} &times; IQR<br><br>
Traditionally the multiplier used is 1.5 &times; IQR. However,<br>
datasets such as the Titanic Fare distribution are highly<br>
skewed. Using 1.5 may label many moderate values as outliers.<br><br>
Therefore a relaxed multiplier (${multiplier} &times; IQR) is used<br>
to focus on more extreme deviations and reduce excessive<br>
outlier detection.<br>`;
  } else {
    explanationHTML += `
Z-Score Method<br><br>
The Z-Score method determines how far a data point lies<br>
from the mean of the dataset in terms of standard deviations.<br><br>
Formula:<br>
z = (x &minus; &mu;) / &sigma;<br><br>
Where:<br>
x = data value<br>
&mu; = mean of dataset<br>
&sigma; = standard deviation<br><br>
Rule used:<br>
|z| > 3 &rarr; Outlier<br><br>
This means the value is more than three standard deviations<br>
away from the mean.<br><br>
This approach works best when the data distribution is<br>
approximately normal.<br>`;
  }

  // Base64 or generic names based on the method
  let imgSrc = `images/${feature.toLowerCase()}_${method.toLowerCase()}.png`;

  let visualHTML = `<div style="margin-top: 20px;">
        <img src="${imgSrc}" onerror="this.src=''; this.alt=''" style="width: 500px; border: 1px solid #aaa;" />
    </div>`;

  previewArea.innerHTML = `<div style="font-family: monospace; font-size: 13px; line-height: 1.4;">
        Selected Feature: ${feature}<br>
        Detection Method: ${method}<br>
        Total Data Points: ${dataPoints}<br>
        Number of Outliers Detected: ${outCount}<br><br>
        Concept Explanation:<br><br>
        <div style="margin-left: 0px;">${explanationHTML}</div>
    </div>${visualHTML}`;
};

window.onOutlierChange = function () {
  window.updateOutlierPreview();
};

window.updateOutlierPreview = function () {
  const previewArea = document.getElementById("outlierPreviewArea");
  const mSelect = document.getElementById("outlierMethodDropdown");
  const fSelect = document.getElementById("outlierFeatureDropdown");
  if (!previewArea || !mSelect || !fSelect) return;

  let method = mSelect.value;
  let feature = fSelect.value;

  let dataPoints = feature === "Age" ? 714 : 891;
  let outCount = 0;
  if (feature === "Age" && method === "IQR") outCount = 1;
  else if (feature === "Age" && method === "Z-Score") outCount = 2;
  else if (feature === "Fare" && method === "IQR") outCount = 34;
  else if (feature === "Fare" && method === "Z-Score") outCount = 20;

  let explanationHTML = "";
  if (method === "IQR") {
    let multiplier = feature === "Fare" ? 4.5 : 2.3;
    explanationHTML += `
IQR (Interquartile Range) Method<br><br>
The IQR method detects outliers by examining the spread<br>
of the middle 50% of the dataset.<br><br>
Formula:<br>
IQR = Q3 &minus; Q1<br><br>
Where:<br>
Q1 = 25th percentile<br>
Q3 = 75th percentile<br><br>
Outlier limits:<br><br>
Lower Bound = Q1 &minus; ${multiplier} &times; IQR<br>
Upper Bound = Q3 + ${multiplier} &times; IQR<br><br>
Traditionally the multiplier used is 1.5 &times; IQR. However,<br>
datasets such as the Titanic Fare distribution are highly<br>
skewed. Using 1.5 may label many moderate values as outliers.<br><br>
Therefore a relaxed multiplier (${multiplier} &times; IQR) is used<br>
to focus on more extreme deviations and reduce excessive<br>
outlier detection.<br>`;
  } else {
    explanationHTML += `
Z-Score Method<br><br>
The Z-Score method determines how far a data point lies<br>
from the mean of the dataset in terms of standard deviations.<br><br>
Formula:<br>
z = (x &minus; &mu;) / &sigma;<br><br>
Where:<br>
x = data value<br>
&mu; = mean of dataset<br>
&sigma; = standard deviation<br><br>
Rule used:<br>
|z| > 3 &rarr; Outlier<br><br>
This means the value is more than three standard deviations<br>
away from the mean.<br><br>
This approach works best when the data distribution is<br>
approximately normal.<br>`;
  }

  // Base64 or generic names based on the method
  let imgSrc = `images/${feature.toLowerCase()}_${method.toLowerCase()}.png`;

  let visualHTML = `<div style="margin-top: 20px;">
        <img src="${imgSrc}" onerror="this.src=''; this.alt=''" style="width: 500px; border: 1px solid #aaa;" />
    </div>`;

  previewArea.innerHTML = `<div style="font-family: monospace; font-size: 13px; line-height: 1.4;">
        Selected Feature: ${feature}<br>
        Detection Method: ${method}<br>
        Total Data Points: ${dataPoints}<br>
        Number of Outliers Detected: ${outCount}<br><br>
        Concept Explanation:<br><br>
        <div style="margin-left: 0px;">${explanationHTML}</div>
    </div>${visualHTML}`;
};
