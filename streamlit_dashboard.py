import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Netflix Dashboard",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🎬 Netflix Content Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("netflix_dashboard_data.csv")

# ---------------- DATA PREPROCESSING ----------------

# Convert is_movie to readable labels
df['type'] = df['is_movie'].map({1: 'Movie', 0: 'TV Show'})

# -------- FIX COUNTRY ISSUE --------
df['country'] = df['country'].fillna('')
df['country_list'] = df['country'].apply(
    lambda x: [c.strip() for c in x.split(',') if c.strip()]
)

# Unique country list
all_countries = sorted({
    country for sublist in df['country_list'] for country in sublist
})

# -------- GENRE COLUMNS --------
exclude_cols = ['is_movie', 'release_year', 'duration_int', 'cluster']
genre_cols = [
    col for col in df.columns
    if col not in exclude_cols and df[col].dtype in ['int64', 'float64']
]

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")

# Type filter
type_filter = st.sidebar.multiselect(
    "Select Type",
    options=df['type'].unique(),
    default=df['type'].unique()
)

# Country filter
country_filter = st.sidebar.multiselect(
    "Select Country",
    options=all_countries,
    default=all_countries
)

# Rating filter
rating_filter = st.sidebar.multiselect(
    "Select Rating",
    options=sorted(df['rating'].dropna().unique()),
    default=sorted(df['rating'].dropna().unique())
)

# Year filter
year_filter = st.sidebar.slider(
    "Select Release Year Range",
    int(df['release_year'].min()),
    int(df['release_year'].max()),
    (int(df['release_year'].min()), int(df['release_year'].max()))
)

# Genre filter
genre_filter = st.sidebar.multiselect(
    "Select Genres",
    options=genre_cols,
    default=genre_cols
)

# ---------------- APPLY FILTERS ----------------

filtered_df = df[
    (df['type'].isin(type_filter)) &
    (df['rating'].isin(rating_filter)) &
    (df['release_year'].between(year_filter[0], year_filter[1]))
]

# -------- SAFE COUNTRY FILTER --------
if len(country_filter) > 0:
    filtered_df = filtered_df[
        filtered_df['country_list'].apply(
            lambda x: any(c in x for c in country_filter)
        )
    ]
else:
    st.warning("⚠️ Please select at least one country")
    filtered_df = filtered_df.iloc[0:0]

# -------- SAFE GENRE FILTER --------
if len(genre_filter) > 0:
    filtered_df = filtered_df[
        filtered_df[genre_filter].sum(axis=1) > 0
    ]
else:
    st.warning("⚠️ Please select at least one genre")
    filtered_df = filtered_df.iloc[0:0]

# -------- STOP IF EMPTY --------
if filtered_df.empty:
    st.error("🚫 No data available for selected filters")
    st.stop()

# ---------------- DISPLAY DATA ----------------

st.subheader("Filtered Dataset")

filtered_df['country_display'] = filtered_df['country_list'].apply(
    lambda x: ", ".join(x)
)

st.dataframe(filtered_df.drop(columns=['country_list']))

# ---------------- KPI METRICS ----------------

st.subheader("Key Insights")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", len(filtered_df))
col2.metric("Movies", filtered_df[filtered_df['type'] == 'Movie'].shape[0])
col3.metric("TV Shows", filtered_df[filtered_df['type'] == 'TV Show'].shape[0])
col4.metric("Countries", len(set(
    c for sublist in filtered_df['country_list'] for c in sublist
)))

# ---------------- CHARTS ----------------

st.subheader("Visual Insights")

col1, col2 = st.columns(2)

# 1️⃣ Content Type Distribution
with col1:
    st.markdown("### Content Type Distribution")
    type_counts = filtered_df['type'].value_counts()
    fig, ax = plt.subplots()
    type_counts.plot(kind='bar', ax=ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    st.pyplot(fig)

# 2️⃣ Release Year Trend
with col2:
    st.markdown("### Release Year Trend")
    year_counts = filtered_df.groupby('release_year').size()
    fig, ax = plt.subplots()
    year_counts.plot(ax=ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    st.pyplot(fig)

# ---------------- SECOND ROW ----------------

col3, col4 = st.columns(2)

# 3️⃣ Top Countries
with col3:
    st.markdown("### Top Countries")
    
    all_country_series = pd.Series(
        [c for sublist in filtered_df['country_list'] for c in sublist]
    )
    
    country_counts = all_country_series.value_counts().head(10)
    
    fig, ax = plt.subplots()
    country_counts.plot(kind='barh', ax=ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    st.pyplot(fig)

# 4️⃣ Rating Distribution
with col4:
    st.markdown("### Rating Distribution")
    rating_counts = filtered_df['rating'].value_counts()
    fig, ax = plt.subplots()
    rating_counts.plot(kind='bar', ax=ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    st.pyplot(fig)

# ---------------- GENRE ANALYSIS ----------------

st.subheader("Genre Analysis")

genre_counts = filtered_df[genre_cols].sum().sort_values(ascending=False)
top_genres = genre_counts.head(10)

fig, ax = plt.subplots()
top_genres.sort_values().plot(kind='barh', ax=ax)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

st.pyplot(fig)
