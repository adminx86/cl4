'''
Data Visualization from Extraction Transformation and Loading (ETL) Process 
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: EXTRACT
# -------------------------------
# Load data from the CSV file
df = pd.read_csv('sales_data.csv')  # Ensure this file is in the same directory

# -------------------------------
# Step 2: TRANSFORM
# -------------------------------
# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract 'month' from date
df['month'] = df['date'].dt.to_period('M')

# Group by month and product, and calculate total revenue
monthly_sales = df.groupby(['month', 'product'])['revenue'].sum().reset_index()

# Convert 'month' to string for plotting (to avoid Seaborn error)
monthly_sales['month'] = monthly_sales['month'].astype(str)

# -------------------------------
# Step 3: LOAD
# -------------------------------
# Save the transformed data (optional step)
monthly_sales.to_csv('monthly_sales_summary.csv', index=False)

# -------------------------------
# Step 4: VISUALIZE
# -------------------------------
# Set seaborn style
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x='month', y='revenue', hue='product', marker='o')

# Add labels and title
plt.title('Monthly Revenue by Product')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
