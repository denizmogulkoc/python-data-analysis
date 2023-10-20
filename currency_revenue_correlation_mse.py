import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Step 1: Data Collection (Assuming you have a CSV file with two columns: 'revenue' and 'exchange_rate')
df = pd.read_csv('revenue_currency.csv')

df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%d/%m/%Y"))

# Step 2: Data Preprocessing
X = df['Currency'].values.reshape(-1, 1)
y = df['Revenue'].values

correlation = df['Currency'].corr(df['Revenue'])

correlation

average_revenue = df['Revenue'][:20].mean()  # Assuming have data for 20 months
average_revenue

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression

# Assuming df contains your data
X = df[['Currency']]  # Features (independent variable)
y = df['Revenue']     # Target (dependent variable)

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model using RMSE
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Calculate RMSE from MSE

# Baseline Model (Assuming it's a simple mean)
y_pred_baseline = [y_train.mean()] * len(y_test)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
rmse_baseline = mse_baseline ** 0.5

# Print RMSE values
print(f"Random Forest RMSE: {rmse}")
print(f"Baseline RMSE: {rmse_baseline}")

# Calculate RMSE delta
rmse_delta = rmse_baseline - rmse

# Create scatter chart
plt.scatter(['Random Forest', 'Baseline'], [rmse, rmse_baseline])
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('RMSE Comparison')
plt.show()

print(f"RMSE Delta: {rmse_delta}")
