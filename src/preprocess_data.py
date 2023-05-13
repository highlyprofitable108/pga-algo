def is_row_valid(row):
    required_columns = ['tour', 'year', 'season', 'event_completed', 'event_name', 'event_id', 'player_name', 'dg_id', 
                        'fin_text', 'round_num', 'course_name', 'course_num', 'course_par', 'round_score', 'sg_putt', 
                        'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'driving_dist', 'driving_acc', 'gir', 
                        'scrambling', 'prox_rgh', 'prox_fw']
    return all(pd.notnull(row[col]) for col in required_columns)

def preprocess_data(data):
    """
    Preprocess the data, including handling missing values, encoding categorical variables,
    and scaling numerical variables.

    :param data: pandas DataFrame containing the golf data
    :return: pandas DataFrame with preprocessed data
    """
    try:
        # Filter out rows that do not contain all required variables
        data = data[data.apply(is_row_valid, axis=1)]

        # Handle missing values
        # Replace missing numerical values with the median value of the column
        numerical_cols = data.select_dtypes(include=np.number).columns.tolist()

        for col in numerical_cols:
            data[col].fillna(data[col].median(), inplace=True)

        # Replace missing categorical values with the mode of the column
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)

        # Encode categorical variables
        # One-hot encode categorical variables with multiple categories
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Scale numerical variables
        # Normalize numerical variables to the range [0, 1]
        for col in numerical_cols:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

        return data

    except Exception as e:
        print(f"Error occurred during data preprocessing: {str(e)}")
        return None

# Advanced Missing Value Imputation: The current function uses simple imputation methods for missing values: median for numerical variables and mode for categorical ones. Depending on your data, you might want to consider more advanced imputation methods. For example, K-Nearest Neighbors (KNN) or Multivariate Imputation by Chained Equations (MICE) for numerical variables, and predictive imputation or multiple imputations for categorical ones.
# 
# Feature Scaling: The current function uses min-max scaling for numerical variables. Depending on your data and model, you might want to consider other scaling methods, such as standard scaling (z-score normalization), log transformation, or others.
# 
# Categorical Variable Encoding: The current function uses one-hot encoding for categorical variables. Depending on the number of categories and their nature, other encoding methods might be more suitable. For example, ordinal encoding for ordinal variables, binary encoding for high cardinality variables, target encoding, etc.
# 
# Outlier Detection and Handling: The current function does not include any specific handling for outliers. Depending on your data and model, you might want to add an outlier detection and handling step. For example, using z-score, IQR, or more advanced methods like DBSCAN or Isolation Forest, and then removing or transforming the outliers.
# 
# Feature Extraction: Depending on your data, there might be opportunities for feature extraction. For example, extracting the year, month, and day from a date variable, calculating the time since a particular event, etc.
# 
# Data Transformation: Depending on your data and model, you might want to apply some data transformations. For example, applying a log transformation to skewed numerical variables, or creating interaction terms for certain variables.
