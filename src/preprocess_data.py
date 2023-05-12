def preprocess_data(data):
    """
    Preprocess the data, including handling missing values, encoding categorical variables,
    and scaling numerical variables.
    
    :param data: pandas DataFrame containing the golf data
    :return: pandas DataFrame with preprocessed data
    """

    # Handle missing values
    # Replace missing numerical values with the median value of the column
    numerical_cols = data.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        data[col].fillna(data[col].median(), inplace=True)

    # Replace missing categorical values with the mode of the column
    for col in data.select_dtypes(include='category'):
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Encode categorical variables
    # One-hot encode categorical variables with multiple categories
    data = pd.get_dummies(data, drop_first=True)

    # Scale numerical variables
    # Normalize numerical variables to the range [0, 1]
    for col in numerical_cols:
        if col in data.columns: # ensure the column still exists
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    return data
