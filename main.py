import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Import your custom functions
from src.preprocess_data import preprocess_data
from src.create_features import create_features
from src.train_model import train_model

def main():
    # Load the datasets
    url1 = 'https://raw.githubusercontent.com/zygmuntz/golf-data/master/golf.csv'
    url2 = 'https://raw.githubusercontent.com/UCD-GW-Nitrate/nitrate.main/master/Project_Data/golf_course_reviews.csv'

    data1 = pd.read_csv(url1)
    data2 = pd.read_csv(url2)

    # Combine the datasets
    combined_data = pd.concat([data1, data2], ignore_index=True)

    # Preprocess the data
    preprocessed_data = preprocess_data(combined_data)

    # Create new features
    feature_data = create_features(preprocessed_data)

    # Separate the feature variables and the target variable
    X = feature_data.drop('target_column_name', axis=1)
    y = feature_data['target_column_name']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    best_model = train_model(X_train, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

if __name__ == "__main__":
    main()
