import pandas as pd
from sklearn.model_selection import train_test_split

# Import your custom functions
from src.preprocess_data import preprocess_data
from src.create_features import create_features
from src.train_model import train_model
from src.evaluate_model import evaluate_model

def main():
    # Load the datasets
    # url1 = 'https://raw.githubusercontent.com/zygmuntz/golf-data/master/golf.csv'
    # url2 = 'https://raw.githubusercontent.com/UCD-GW-Nitrate/nitrate.main/master/Project_Data/golf_course_reviews.csv'

    data1 = pd.read_csv('data/ASA All PGA Raw Data - Tourn Level.csv')
    # data2 = pd.read_csv(url2)

    # Combine the datasets
    # combined_data = pd.concat([data1, data2], ignore_index=True)

    # Preprocess the data
    # preprocessed_data = preprocess_data(combined_data)

    # Create new features
    feature_data = create_features(data1)

    # Separate the feature variables and the target variable
    X = feature_data.drop('strokes_gained', axis=1)  # Replace 'strokes_gained' with your target column
    y = feature_data['strokes_gained']  # Replace 'strokes_gained' with your target column

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    models = train_model(X_train, y_train)

    # Evaluate the model
    mae, rmse = evaluate_model(models, X_test, y_test)

    # Print the evaluation metrics
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)

if __name__ == "__main__":
    main()
