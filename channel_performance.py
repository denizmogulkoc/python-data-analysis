import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('black_friday.csv')

# Display the first few rows to understand the structure
print(df.head())

df.set_index(df.columns[0], inplace=True)

df = df.dropna()

df['Revenue'] = df['Revenue'].astype(float)
df['Ecommerce Conversion Rate'] = df['Ecommerce Conversion Rate'].astype(float)
df['Sessions'] = df['Sessions'].astype(float)

# Calculate R, F, and M scores
# R: Recency (1: Least recent, 4: Most recent)
df['R'] = pd.qcut(df['Sessions'], q=4, labels=False, precision=0)

# F: Frequency (1: Least frequent, 4: Most frequent)
df['F'] = pd.qcut(df['Ecommerce Conversion Rate'], q=4, labels=False, precision=0)

# M: Monetary (1: Least revenue, 4: Most revenue)
df['M'] = pd.qcut(df['Revenue'], q=4, labels=False, precision=0)

# Calculate RFM score
df['Channel_Score'] = df['R'] + df['F'] + df['M']

print(df.head())

top_customers = df[df['Channel_Score'] == df['Channel_Score'].max()]

# Display the top customers
print(top_customers)

channel_scores = df['Channel_Score']

plt.figure(figsize=(10, 6))
plt.bar(df.index, channel_scores, color='red')
plt.xlabel('Row Index')
plt.ylabel('Channel Score')
plt.title('Channel Scores')

# Set the x-axis labels to be displayed vertically
plt.xticks(rotation='vertical')

plt.show()
