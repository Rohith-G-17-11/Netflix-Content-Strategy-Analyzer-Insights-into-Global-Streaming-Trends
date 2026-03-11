import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("netflix_titles.csv", encoding="utf-8")

print("Initial Shape:", df.shape)
print("\nMissing Values Before:\n", df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Clean column names
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

#Fix text Encoding
text_cols = ['title', 'director', 'cast', 'country', 'listed_in', 'description']

for col in text_cols:
    df[col] = df[col].apply(
        lambda x: x.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
        if isinstance(x, str) else x
    )


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

#new features include content type, year and month added,and content age.

# Binary feature for content type
df['is_movie'] = df['type'].map({'Movie': 1, 'TV Show': 0})

# Extract year and month
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Content age
df['content_age'] = 2026 - df['release_year']

# Clean genres properly
df['listed_in'] = df['listed_in'].str.lower()

df['listed_in'] = df['listed_in'].apply(
    lambda x: ','.join([i.strip() for i in x.split(',')]) 
    if isinstance(x, str) else x
)

# One-hot encode genres
genre_dummies = df['listed_in'].str.get_dummies(sep=',')

# Remove duplicate columns if any
genre_dummies = genre_dummies.loc[:, ~genre_dummies.columns.duplicated()]

df = pd.concat([df, genre_dummies], axis=1)

le_rating = LabelEncoder()
le_country = LabelEncoder()

df['rating_encoded'] = le_rating.fit_transform(df['rating'])
df['country_encoded'] = le_country.fit_transform(df['country'])

# Save encoding mapping for app usage
rating_mapping = { 
    str(k): int(v) 
    for k, v in zip(le_rating.classes_, le_rating.transform(le_rating.classes_))
}

country_mapping = { 
    str(k): int(v) 
    for k, v in zip(le_country.classes_, le_country.transform(le_country.classes_))
}


with open("encoding_mappings.json", "w") as f:
    json.dump({
        "rating_mapping": rating_mapping,
        "country_mapping": country_mapping
    }, f, indent=4)


# Dropping initial text fields
df.drop(columns=['type', 'rating', 'country', 'listed_in', 'date_added'], inplace=True)

# Final check
print("\nFinal Shape:", df.shape)
print("\nMissing Values After:\n", df.isnull().sum())

# Save cleaned dataset
df.to_csv("netflix_cleaned.csv", index=False)

print("\nCleaning Completed Successfully")
