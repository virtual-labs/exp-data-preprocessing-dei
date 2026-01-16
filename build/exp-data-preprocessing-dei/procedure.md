In this experiment we will perform data preprocessing and feature engineering on the Titanic dataset by handling missing values, encoding categorical variables, normalizing numerical features, creating new meaningful features, and visualizing the data, in order to improve data quality and prepare it for effective machine learning model development.

**Step 1**: **Importing Required Libraries**: Required Python libraries are imported at the beginning of the program. These include NumPy and Pandas for data handling, Matplotlib and Seaborn for data visualization, and Scikit-learn utilities for preprocessing, model preparation, and evaluation.

**Step 2**: **Loading the Dataset**: The Titanic dataset is loaded from a CSV file using the `read_csv()` function of Pandas. The dataset is stored in a DataFrame. After loading, the dataset contains a fixed number of rows (891) and columns (12) representing passenger information.

**Step 3**: **Displaying Initial Records**: The first five rows of the dataset are displayed using the `head()` function to obtain an initial understanding of the data values and structure.

**Step 4**: **Dataset Analysis**: Basic dataset analysis is performed using the `info()` function to examine:
- Column names
- Data types of each feature
- Number of non-null values
This step helps in identifying numerical and categorical attributes and detecting missing values.

**Step 5**: **Identifying Missing Values**: Missing values in the dataset are identified by checking null counts for each column. This helps determine which attributes require missing value treatment before further processing.

**Step 6**: **Handling Missing Values**: Missing values are handled as follows:
- The **Age** attribute is filled using the mean value of the column.
- The **Embarked** attribute is filled using the most frequent value (mode).
- The **Cabin** column is dropped due to a large number of missing values.
After this step, a cleaned dataset (`df_clean`) is created with no null values.

**Step 7**: **Normalization of Numerical Features**: Numerical features such as **Age** and **Fare** are normalized using `StandardScaler`. Normalization brings all numerical features to a common scale and prevents features with larger values from dominating the learning process.

**Step 8**: **Encoding Categorical Variables**: Categorical features are converted into numerical form:
- The **Sex** attribute is encoded using Label Encoding.
- The **Embarked** attribute is also encoded numerically.
Encoding is necessary to make categorical data compatible with machine learning algorithms.

**Step 9**: **Data Visualization**: Data visualization is performed to understand feature distributions and relationships:
- Count plots are used for categorical variables such as **Sex** and **Survival**.
- Distribution plots and box plots are used for numerical variables such as **Age** and **Fare**.
Visualization helps in identifying patterns, trends, and potential outliers in the data.

**Step 10**: **Feature Engineering – Creating New Features**: New meaningful features are created to improve model learning:
- **FamilySize** is created using the formula: `FamilySize = SibSp + Parch + 1`
- **IsAlone** is created where:
    - `IsAlone = 1` if **FamilySize** equals 1
    - `IsAlone = 0` otherwise
These features capture family-related passenger information.

**Step 11**: **Adding Engineered Features to Dataset**: The newly created features **FamilySize** and **IsAlone** are added to the cleaned dataset (`df_clean`) and displayed to verify correct feature creation.

**Step 12**: **Final Dataset Verification**: The final cleaned and preprocessed dataset is examined using descriptive statistics to verify:
- No missing values remain
- All categorical variables are encoded
- Numerical features are normalized
- New engineered features are successfully added