
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

"""**Load and View datasets**"""

# Load data
train_data = pd.read_csv('/content/train.csv')
test_data = pd.read_csv('/content/test.csv')
sample_submission = pd.read_csv('/content/sample_submission.csv')

# View the data
train_data.head()
test_data.head()

"""**Preprocessing the Data**"""

# Feature selection based on the columns available in train_data
X = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
y = train_data['SalePrice']

"""**Split the Data**"""

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

"""**Initialising and training**"""

model = LinearRegression()
model.fit(X_train, y_train)

"""**Evaluation**"""

# Predict on validation data
y_pred = model.predict(X_val)

# Calculate MSE
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')

"""**Prepare and Predict Test data**"""

# Prepare the test data
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]

# Predict on test data
test_predictions = model.predict(X_test)

"""**Preparing  and Downloading Submission File**"""

# Prepare the submission file
submission = sample_submission.copy()
submission['SalePrice'] = test_predictions
submission.to_csv('house_price_predictions.csv', index=False)