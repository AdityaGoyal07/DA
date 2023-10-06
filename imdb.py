import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer

data = pd.read_csv("C:/Users/adity/Desktop/IMDb_Movies_India.csv",encoding='latin1')

# Data preprocessing
# Drop rows with missing ratings
data = data.dropna()

# Feature engineering
# Convert genres, directors, and actors into binary columns
mlb = MultiLabelBinarizer()
genre_df = pd.DataFrame(mlb.fit_transform(data['Genre']), columns=mlb.classes_)
director_df = pd.DataFrame(mlb.fit_transform(data['Director']), columns=mlb.classes_)
actors_df = pd.DataFrame(mlb.fit_transform(data['Actor 1']), columns=mlb.classes_)

# Combine the binary feature dataframes with the original dataframe
data = pd.concat([data, genre_df, director_df, actors_df], axis=1)

# Select relevant features and target variable
features = ['Year', 'Duration', 'Votes'] + list(genre_df.columns) + list(director_df.columns) + list(actors_df.columns)
X = data[features]
y = data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# You can now use this model to predict the rating of a new movie by providing its features in the same format as X.
