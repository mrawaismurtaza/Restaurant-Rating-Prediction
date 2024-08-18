import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Importing the model
import joblib

st.title('Restaurant Ratings Prediction App')


# Describing Data
st.subheader('Restarant Raw Data')
restaurants = pd.read_csv('Dataset.csv')
st.write(restaurants)


# Removing Restraurant Id column
restaurants = restaurants.drop(columns=['Restaurant ID'])



# Label Encoding Multiple Columns
label_encoder = LabelEncoder()
columns_encode = ['Restaurant Name','City','Address','Locality','Locality Verbose','Cuisines','Currency','Rating color','Rating text']

for column in columns_encode:
    restaurants[column] = label_encoder.fit_transform(restaurants[column])

columns=['Has Table booking','Has Online delivery','Is delivering now','Switch to order menu']
for column in columns:
    restaurants[column] = label_encoder.fit_transform(restaurants[column])


st.subheader('After Label Encoding')
st.write(restaurants.head(5))


#Splitting data into Features and Labels
X = restaurants.drop(columns=['Aggregate rating'])
y = restaurants['Aggregate rating']



#Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

st.subheader('Splitted Data')
st.write(X_train.head())
st.write(y_train.head())


# Load the Model
try:
    model = joblib.load('random_forest_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found.")
except joblib.exceptions.ByteStreamError as e:
    print(f"Error reading model file: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

#Prediction on testing data
y_pred = model.predict(X_test)


#Performance and Error Checking
mse_perf = mean_squared_error(y_test, y_pred)
r2_perf = r2_score(y_test, y_pred)

print("Mean Squared Error :", mse_perf)
print("R-squared :", r2_perf)



# Plotting predicted ratings from testing data
st.subheader('Predicted ratings from testing data')
fig1 = plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, color='b', label='Actual Aggregate Ratings', alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, color='r', label='Predicted Aggregate Ratings', alpha=0.5)
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Index')
plt.ylabel('Rating')
plt.legend()
plt.show()
st.pyplot(fig1)


# Plotting predicted ratings

fig2 = plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, color='blue', label='Actual Aggregate Ratings')
plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted Aggregate Ratings')
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Index')
plt.ylabel('Rating')
plt.legend()
plt.show()
st.pyplot(fig2)


# Get feature importances
importances = model.feature_importances_
feature_names = X_train.columns 

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)



#Displaying Most important Feature

st.subheader('Most important Feature')
fig3 = plt.figure(figsize=(30, 15))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Importance Value')
plt.ylabel('Importance Value')
plt.title('Feature Importances')
plt.gca().invert_yaxis() 
plt.show()

st.pyplot(fig3)