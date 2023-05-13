def create_features(data):
    """
    Create new features from the existing data.
    
    :param data: pandas DataFrame containing the golf data
    :return: pandas DataFrame with new features added
    """
    try:
        # Checking if the data is None or empty
        if data is None or data.empty:
            raise ValueError("Input data is None or empty")
        
        # Handle missing values before creating new features
        numerical_cols = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'driving_dist', 'driving_acc', 'gir', 'scrambling', 'prox_rgh', 'prox_fw']
        for col in numerical_cols:
            data[col].fillna(data[col].median(), inplace=True)

        categorical_cols = ['tour', 'year', 'season', 'event_completed', 'event_name', 'event_id', 'player_name', 'dg_id', 'fin_text', 'round_num', 'course_name', 'course_num', 'course_par', 'round_score']
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)

        selected_data = data.copy()  # Create a copy of the original data

        # Create interaction features between variables
        selected_data['sg_putt_gir'] = selected_data['sg_putt'] * selected_data['gir']
        selected_data['sg_app_prox_fw'] = selected_data['sg_app'] * selected_data['prox_fw']
        selected_data['sg_ott_driving_dist'] = selected_data['sg_ott'] * selected_data['driving_dist']
        
        # Create polynomial features
        poly = PolynomialFeatures(2, interaction_only=True)
        poly_data = poly.fit_transform(selected_data[numerical_cols])
        target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(numerical_cols,p) for p in poly.powers_]]
        poly_data = pd.DataFrame(poly_data, columns=target_feature_names)

        # Merge the polynomial features with the original data
        selected_data = pd.concat([selected_data, poly_data], axis=1)
        
        # Calculate the average performance per season
        selected_data['avg_sg_season'] = selected_data.groupby('season')['sg_total'].transform('mean')
        
        # Convert year to "years since start of data"
        selected_data['years_since_start'] = selected_data['year'] - selected_data['year'].min()

        # Add player performance per course
        selected_data['avg_sg_player_course'] = selected_data.groupby(['player_name', 'course_name'])['sg_total'].transform('mean')

        # Handle blank values by replacing them with appropriate missing value representation
        selected_data.replace('', np.nan, inplace=True)

        # Perform feature selection (example using Lasso regularization)
        X = selected_data.drop('sg_total', axis=1)  # Remove the target variable
        y = selected_data['sg_total'].values.reshape(-1, 1)  # Separate the target column
        
        # Replace the missing values in X and y with appropriate representation
        X = X.fillna(X.median())
        y = y.fillna(y.median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_scaled, y)
        feature_coef = pd.Series(lasso.coef_, index=X.columns)
        selected_features = feature_coef[feature_coef != 0].index.tolist()

        # Include only the selected features in the result
        selected_data = selected_data[selected_features]

        return selected_data

    except Exception as e:
        print(f"Error occurred during feature creation: {str(e)}")
        return None

# Incorporate weather data: As suggested in the function, you can fetch historical weather data based on the date and location of each event, and add this data as new features. Weather conditions like temperature, precipitation, and wind speed might have significant effects on golf games.
# 
# Incorporate player-specific data: If you have access to player-specific data, such as player's age, physical condition, or historical performance, you might consider adding these as new features as well.
# 
# Use more advanced feature selection methods: The current function uses Lasso regression for feature selection, which is a simple and commonly used method. But depending on your specific needs, you might want to use more sophisticated methods, such as Recursive Feature Elimination, SelectFromModel, or others.
# 
# Hyperparameter tuning: The alpha parameter for the Lasso model is currently set to a fixed value of 0.1. You might want to use GridSearchCV or other hyperparameter tuning methods to find the optimal value.
# 
# Automate feature engineering: Instead of manually creating interaction features and polynomial features, you could use automated feature engineering tools like Featuretools.
# 
# Handling of missing values: Currently, missing values are replaced with appropriate representations (like the mean). You might want to explore more sophisticated imputation methods, such as KNN imputation or multivariate imputation by chained equations (MICE).
