import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import mlflow

# Allow old file store
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

# 1. Generate sample data
np.random.seed(42)
n = 500
units = np.random.randint(50, 500, n)
appliances = np.random.randint(1, 10, n)
months = np.random.randint(1, 12, n)
bill = units * 6.5 + appliances * 50 + np.random.randint(-200, 200, n)

df = pd.DataFrame({
    'Units': units,
    'Num_Appliances': appliances,
    'Month': months,
    'Bill_Amount': bill
})

X = df[['Units', 'Num_Appliances', 'Month']]
y = df['Bill_Amount']

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Build Neural Network
mlflow.set_experiment("Electricity_Bill_Prediction")

with mlflow.start_run():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, validation_split=0.1, verbose=1)
    
    loss, mae = model.evaluate(X_test, y_test)
    print(f"MAE: {mae}")
    
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("loss", loss)

    model.save('electricity_model.keras')
    print("Model saved as electricity_model.keras")

# 5. Save comparison
pd.DataFrame({'Actual': y_test.values[:10], 'Predicted': model.predict(X_test[:10]).flatten()}).to_csv('model_comparison.csv', index=False)