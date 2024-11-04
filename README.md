House Price Prediction using Linear Regression
This project implements a linear regression model to predict house prices based on key features such as the living area (square footage), number of bedrooms, and bathrooms. It is designed to be run in a Google Colab notebook with provided training, test, and sample submission data.

Project Overview
The model uses a straightforward linear regression approach to predict house prices by training on labeled housing data. The selected features include:

GrLivArea: Above-ground living area in square feet
BedroomAbvGr: Number of bedrooms above ground
FullBath: Number of full bathrooms
HalfBath: Number of half bathrooms
Dataset
The project includes three datasets:

train.csv: Labeled data for training the model
test.csv: Unlabeled data for making predictions
sample_submission.csv: Sample file format for the submission
Getting Started
Prerequisites
To run this project, you'll need:

Python 3.x
Libraries: pandas, sklearn
Running the Model in Google Colab
Upload the files (train.csv, test.csv, sample_submission.csv) to Colab.

Run the program

Results
The model outputs a Mean Squared Error (MSE) score on the validation set, providing a performance metric to assess accuracy.

Future Improvements
Consider experimenting with additional features, regularization techniques (e.g., Ridge or Lasso regression), or tuning hyperparameters for improved accuracy.
