In this experiment we will perform data preprocessing and feature engineering on the Titanic dataset by handling missing values, encoding categorical variables, normalizing numerical features, creating new meaningful features, and visualizing the data, in order to improve data quality and prepare it for effective machine learning model development.

### Step 1: Importing Required Libraries
The required Python libraries are imported at the beginning of the program. These include NumPy and Pandas for data manipulation and handling, Matplotlib and Seaborn for data visualization, and Scikit-learn utilities such as `LabelEncoder`, `OrdinalEncoder`, `StandardScaler`, `MinMaxScaler`, `RobustScaler`, and z-score functions for preprocessing operations like encoding, feature scaling, and outlier detection.
Additionally, IPython display utilities (HTML display) and ipywidgets are used to build an interactive simulation environment, enabling dynamic user inputs and real-time visualization of preprocessing steps.

### Step 2: Loading the Dataset
The Titanic dataset is loaded from a CSV file using the `read_csv()` function of the Pandas library. The dataset is stored in a DataFrame structure for further analysis. After loading, the dataset contains 891 rows and multiple columns representing passenger-related attributes.

### Step 3: Displaying Initial Records
The first few rows of the dataset are displayed using the `head()` function. This provides an initial overview of the dataset and helps in understanding the structure and values of the attributes.

### Step 4: Dataset Analysis
Basic dataset analysis is performed using the `info()` function to examine:
- Column names
- Data types of each attribute
- Number of non-null values in each column

This step helps distinguish between numerical and categorical variables and provides an initial indication of missing values present in the dataset.

#### Displaying Descriptors
In addition to structural information, descriptive statistics of the dataset are examined using statistical summary functions. These descriptors provide insights into measures such as mean, standard deviation, minimum, maximum, and quartile values, which help in understanding the distribution of numerical attributes.

### Step 5: Identifying Missing Values
Missing values in the dataset are identified by calculating the number of null entries present in each column. This helps determine which attributes require missing value treatment before further preprocessing steps are applied.

### Step 6: Handling Missing Values (Numerical Attributes)
Missing values in numerical attributes are handled using statistical imputation techniques. The Age attribute is treated using different imputation methods such as:
- Mean imputation
- Median imputation
- Mode imputation
- Constant value imputation

These methods replace missing values with meaningful statistical estimates so that the dataset remains complete for further analysis.

### Step 7: Handling Missing Values (Categorical Attributes)
Missing values in categorical attributes are handled using the most frequent value method. The category that occurs most frequently within the feature is identified and used to replace missing entries. This method helps maintain the natural distribution of categorical data while ensuring dataset completeness.

### Step 8: Encoding Categorical Variables
Categorical variables are transformed into numerical representations to make them compatible with machine learning algorithms. The following encoding techniques are applied depending on the nature of the categorical attribute:
- **Label Encoding** for binary categorical variables
- **One-Hot Encoding** for nominal variables without inherent ordering
- **Ordinal Encoding** for categorical attributes that possess a natural ranking

This step ensures that categorical features can be effectively used during model training and analysis.

### Step 9: Feature Scaling (Normalization)
Numerical attributes such as Age and Fare are scaled to a comparable range using feature scaling techniques. Scaling ensures that features with larger magnitudes do not dominate the learning process.
The following scaling techniques are explored:
- **StandardScaler** (Standardization)
- **MinMaxScaler** (Min-Max Normalization)
- **RobustScaler** (Scaling using median and interquartile range)

Feature scaling improves numerical stability and helps machine learning algorithms converge more efficiently.

### Step 10: Outlier Detection
Outliers present in numerical attributes are identified using statistical techniques such as:
- **Z-Score Method**
- **Interquartile Range (IQR) Method**

These techniques help detect extreme values that significantly deviate from the majority of the dataset. Identifying outliers assists in understanding unusual patterns and improving data reliability.

### Step 11: Feature Engineering
Feature engineering is performed to generate new informative attributes from the existing dataset. This step involves transforming or combining relevant attributes to create features that better represent relationships within the data. Such engineered features enhance the dataset’s ability to capture meaningful patterns during analysis.

### Step 12: Data Visualization and Analysis
Data visualization techniques are used to explore the dataset and analyze feature distributions. Graphical representations such as histograms, box plots, and categorical plots are used to study the distribution of attributes, identify patterns, and observe the presence of anomalies or outliers.
Visualization also helps validate preprocessing steps such as scaling, encoding, and missing value handling.

### Step 13: Final Dataset Verification { raw vs processed }
The final preprocessed dataset is examined using descriptive statistics and summary functions to ensure that:
- Missing values have been properly handled
- Categorical variables are correctly encoded
- Numerical attributes are scaled appropriately
- Engineered features are successfully generated
- Outliers have been analyzed

This step confirms that the dataset is clean, consistent, and ready for further machine learning tasks.