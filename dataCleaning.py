import pandas as pd
df = pd.read_csv("netflix_titles.csv")

print("Initial Shape:", df.shape)
print("\nMissing Values Before:\n", df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Clean column names
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

# Handle missing values
df['director'].fillna("Unknown", inplace=True)
df['cast'].fillna("Unknown", inplace=True)
df['country'].fillna("Unknown", inplace=True)
df['rating'].fillna("Not Rated", inplace=True)

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df.dropna(subset=['duration'], inplace=True)

# Clean duration
df[['duration_int', 'duration_type']] = df['duration'].str.split(" ", expand=True)
df['duration_int'] = pd.to_numeric(df['duration_int'], errors='coerce')

# Extract year and month
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Final check
print("\nFinal Shape:", df.shape)
print("\nMissing Values After:\n", df.isnull().sum())

# Save cleaned dataset
df.to_csv("netflix_cleaned.csv", index=False)

print("\nCleaning Completed Successfully")
