import requests
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Import your custom functions
from src.preprocess_data import preprocess_data
from src.create_features import create_features
from src.train_model import train_model
from src.evaluate_model import evaluate_model

def fetch_data(tour, year):
    url = f"https://feeds.datagolf.com/historical-raw-data/rounds?tour={tour}&event_id=all&year={year}&file_format=csv&key=195c3cb68dd9f46d7feaafc4829c"
    response = requests.get(url)
    response.raise_for_status()  # Check for any request errors

    return response.content

def combine_csv_files(tours, start_year, end_year):
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    combined_file_path = os.path.join(data_dir, "combined_data.csv")

    with open(combined_file_path, mode='w', newline='') as combined_file:
        writer = csv.writer(combined_file)
        writer.writerow(['Tour', 'Year', 'Data'])  # Write header row

        for tour in tours:
            for year in range(start_year, end_year + 1):
                csv_data = fetch_data(tour, year).decode('utf-8').strip().split('\n')[1:]  # Remove header row from each CSV
                writer.writerows(csv.reader(csv_data))

    return combined_file_path

def main():
    tours = ['pga', 'kft', 'euro']
    start_year = 2017
    end_year = 2023  # Update to the current year if desired

    combined_csv_path = combine_csv_files(tours, start_year, end_year)
    print(f"Combined CSV file created: {combined_csv_path}")

    # Load the combined dataset
    data = pd.read_csv(combined_csv_path)

    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Create new features
    feature_data = create_features(preprocessed_data)

    # Separate the feature variables and the target variable
    X = feature_data.drop('sg_total', axis=1)  # Replace 'strokes_gained' with your target column
    y = feature_data['sg_total']  # Replace 'strokes_gained' with your target column

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