import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

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

    # Train the model
    best_model = train_model(X, y)

    # Print the best parameters
    print("Best parameters:", best_model.best_params_)

if __name__ == "__main__":
    main()
