import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Netflix Dashboard",
    layout="wide",
    page_icon="🎬"
)

# ---------------- GLOBAL STYLE ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --red:   #E50914;
    --red2:  #b81d24;
    --dark:  #141414;
    --dark2: #1f1f1f;
    --dark3: #2a2a2a;
    --muted: #888;
    --light: #e5e5e5;
    --white: #ffffff;
}

.stApp { background-color: var(--dark); }
.block-container { padding: 2rem 3rem 4rem; }

html, body, [class*="css"], p, span, div, label {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--light);
}

h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.2rem !important;
    letter-spacing: 3px;
    color: var(--white) !important;
}
h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 2px;
    color: var(--white) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--dark2) !important;
    border-right: 1px solid var(--dark3);
}
[data-testid="stSidebar"] * { color: var(--light) !important; }

[data-testid="stMetric"] {
    background: var(--dark2);
    border: 1px solid var(--dark3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: var(--white) !important; font-size: 1.6rem !important; font-weight: 600 !important; }

[data-testid="stTabs"] button {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 2px;
    color: var(--muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--red) !important;
    border-bottom: 2px solid var(--red) !important;
}

.stButton > button {
    background: var(--red) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 2px !important;
    padding: 0.55rem 1.6rem !important;
    transition: background 0.2s;
}
.stButton > button:hover { background: var(--red2) !important; }

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background-color: var(--dark2) !important;
    border: 1px solid var(--dark3) !important;
    color: var(--light) !important;
    border-radius: 6px !important;
}

[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

hr { border-color: var(--dark3) !important; }

.rec-card {
    background: var(--dark2);
    border: 1px solid var(--dark3);
    border-left: 3px solid var(--red);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.rec-title { font-weight: 600; font-size: 0.95rem; color: var(--white); }
.rec-meta  { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
.rec-badge {
    background: var(--red);
    color: var(--white);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
    margin-left: 1rem;
}

.section-label {
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 0.3rem;
    font-weight: 600;
}

.cluster-pill {
    display: inline-block;
    background: var(--dark3);
    border: 1px solid var(--red);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    color: var(--red);
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──
mpl.rcParams.update({
    "figure.facecolor":  "#1f1f1f",
    "axes.facecolor":    "#1f1f1f",
    "axes.edgecolor":    "#2a2a2a",
    "axes.labelcolor":   "#888",
    "xtick.color":       "#888",
    "ytick.color":       "#888",
    "text.color":        "#e5e5e5",
    "grid.color":        "#2a2a2a",
    "grid.linestyle":    "--",
    "axes.grid":         True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
})
RED_PALETTE = ["#E50914", "#b81d24", "#831010", "#4a0a0a"]

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_dashboard_data.csv")
    df['type'] = df['is_movie'].map({1: 'Movie', 0: 'TV Show'})
    df['country'] = df['country'].fillna('')
    df['country_list'] = df['country'].apply(
        lambda x: [c.strip() for c in x.split(',') if c.strip()]
    )
    return df

df = load_data()

exclude_cols = ['is_movie', 'release_year', 'duration_int', 'cluster']
genre_cols = [
    col for col in df.columns
    if col not in exclude_cols and df[col].dtype in ['int64', 'float64']
]
all_countries = sorted({c for sublist in df['country_list'] for c in sublist})

@st.cache_data
def get_genre_matrix(df, genre_cols):
    return df[genre_cols].values.astype(float)

genre_matrix = get_genre_matrix(df, genre_cols)

# ---------------- HEADER ----------------
st.markdown("<h1>🎬 Netflix Content Dashboard</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("### Filters")
type_filter    = st.sidebar.multiselect("Type",    options=df['type'].unique(), default=df['type'].unique())
country_filter = st.sidebar.multiselect("Country", options=all_countries, default=all_countries)
rating_filter  = st.sidebar.multiselect("Rating",  options=sorted(df['rating'].dropna().unique()), default=sorted(df['rating'].dropna().unique()))
year_filter    = st.sidebar.slider("Release Year",
                                    int(df['release_year'].min()), int(df['release_year'].max()),
                                    (int(df['release_year'].min()), int(df['release_year'].max())))
genre_filter   = st.sidebar.multiselect("Genres", options=genre_cols, default=genre_cols)

# ---------------- FILTER ----------------
if not type_filter:
    st.warning("⚠️ Select at least one content type"); filtered_df = df.iloc[0:0]
else:
    filtered_df = df[
        df['type'].isin(type_filter) &
        df['rating'].isin(rating_filter) &
        df['release_year'].between(*year_filter)
    ]

if country_filter:
    filtered_df = filtered_df[filtered_df['country_list'].apply(lambda x: any(c in x for c in country_filter))]
else:
    st.warning("⚠️ Select at least one country"); filtered_df = filtered_df.iloc[0:0]

valid_genres = [g for g in genre_filter if g in filtered_df.columns]
if valid_genres:
    filtered_df = filtered_df[filtered_df[valid_genres].sum(axis=1) > 0]
else:
    st.warning("⚠️ Select at least one genre"); filtered_df = filtered_df.iloc[0:0]

if filtered_df.empty:
    st.error("🚫 No data for selected filters"); st.stop()

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📊  Dashboard", "🎯  Recommendations"])

# ===================== TAB 1 =====================
with tab1:

    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Titles", len(filtered_df))
    k2.metric("Movies",       (filtered_df['type'] == 'Movie').sum())
    k3.metric("TV Shows",     (filtered_df['type'] == 'TV Show').sum())
    k4.metric("Countries",    len({c for sub in filtered_df['country_list'] for c in sub}))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Content Breakdown</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Content Type")
        type_counts = filtered_df['type'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(type_counts, labels=type_counts.index, autopct='%1.0f%%', startangle=90,
               wedgeprops=dict(width=0.35), pctdistance=0.75,
               colors=["#E50914", "#444"])
        ax.set(aspect="equal"); plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with c2:
        st.markdown("### Release Year Trend")
        year_counts = filtered_df.groupby('release_year').size()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(year_counts.index, year_counts.values, color="#E50914", linewidth=2)
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.15, color="#E50914")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)); plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.markdown('<div class="section-label">Distribution</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### Top 10 Countries")
        country_series = pd.Series([c for sub in filtered_df['country_list'] for c in sub])
        country_counts = country_series.value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(country_counts.index[::-1], country_counts.values[::-1], color="#E50914")
        for i, v in enumerate(country_counts.values[::-1]):
            ax.text(v + 0.5, i, str(int(v)), va='center', fontsize=8, color="#888")
        plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    with c4:
        st.markdown("### Rating Distribution")
        rating_counts = filtered_df['rating'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(rating_counts.index, rating_counts.values,
               color=[RED_PALETTE[i % len(RED_PALETTE)] for i in range(len(rating_counts))])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xticks(rotation=30, ha='right', fontsize=8); plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.markdown('<div class="section-label">Genres</div>', unsafe_allow_html=True)
    st.markdown("### Top 10 Genres")
    genre_counts = filtered_df[genre_cols].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], color="#E50914")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Genre Trend Over Years")
    year_genre = filtered_df.groupby('release_year')[genre_cols].sum()
    sel = [g for g in genre_filter if g in year_genre.columns]
    if sel:
        if len(sel) > 5:
            sel = list(year_genre[sel].sum().sort_values(ascending=False).head(5).index)
        trend_colors = ["#E50914", "#ff6b6b", "#ffa07a", "#ffd700", "#90ee90"]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for i, g in enumerate(sel):
            ax.plot(year_genre.index, year_genre[g], label=g.title(),
                    color=trend_colors[i % 5], linewidth=1.8)
        ax.legend(fontsize=8); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    st.markdown("### Rating vs Content Type")
    rating_type = pd.crosstab(filtered_df['rating'], filtered_df['type'])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    rating_type.plot(kind='bar', stacked=True, ax=ax, color=["#E50914", "#444"])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=30, ha='right', fontsize=8); plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown("### Filtered Titles")
    disp = filtered_df.copy()
    disp['country_display'] = disp['country_list'].apply(lambda x: ", ".join(x))
    st.dataframe(disp.drop(columns=['country_list']), use_container_width=True, height=320)


# ===================== TAB 2 =====================
with tab2:

    st.markdown('<div class="section-label">Find Similar Titles</div>', unsafe_allow_html=True)
    st.markdown("### Pick a title — we'll find what's similar")
    st.markdown("<br>", unsafe_allow_html=True)

    col_sel, col_n, col_btn = st.columns([4, 1, 1])

    with col_sel:
        all_titles = sorted(df['title'].dropna().unique())
        selected_title = st.selectbox("Title", options=all_titles, label_visibility="collapsed")

    with col_n:
        top_n = st.number_input("Results", min_value=3, max_value=20, value=8, step=1)

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("Recommend →", use_container_width=True)

    st.markdown("---")

    def get_recommendations(title, top_n=8):
        row = df[df['title'] == title]
        if row.empty:
            return None, None, None
        idx = row.index[0]
        cluster = df.loc[idx, 'cluster']
        vec = genre_matrix[idx].reshape(1, -1)
        mask = (df['cluster'] == cluster) & (df.index != idx)
        cand_idx = df[mask].index.tolist()
        if not cand_idx:
            return pd.DataFrame(), cluster, idx
        sims = cosine_similarity(vec, genre_matrix[cand_idx])[0]
        recs = df.loc[cand_idx].copy()
        recs['score'] = (sims * 100).round(1)
        return recs.sort_values('score', ascending=False).head(top_n), cluster, idx

    if go:
        recs, cluster_id, src_idx = get_recommendations(selected_title, top_n)

        if recs is None:
            st.error("Title not found.")
        else:
            src = df.loc[src_idx]
            src_genres = [g.title() for g in genre_cols if src[g] == 1]

            # Selected title card
            st.markdown(
                f"""
                <div style="background:#1f1f1f;border:1px solid #2a2a2a;border-left:4px solid #E50914;
                            border-radius:10px;padding:1.1rem 1.4rem;margin-bottom:1.4rem;">
                    <div style="font-size:1.25rem;font-weight:700;color:#fff;">{src['title']}</div>
                    <div style="font-size:0.8rem;color:#888;margin-top:4px;">
                        {'Movie' if src['is_movie'] == 1 else 'TV Show'} &nbsp;·&nbsp;
                        {int(src['release_year'])} &nbsp;·&nbsp;
                        {src['rating']} &nbsp;·&nbsp;
                        <span class="cluster-pill">Cluster {int(cluster_id)}</span>
                    </div>
                    <div style="margin-top:8px;font-size:0.78rem;color:#aaa;">
                        {' &nbsp;·&nbsp; '.join(src_genres[:6]) if src_genres else 'No genre tags'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if isinstance(recs, pd.DataFrame) and recs.empty:
                st.warning("No similar titles found in this cluster.")
            else:
                st.markdown(f'<div class="section-label">Top {len(recs)} Recommendations</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                for _, row in recs.iterrows():
                    genres_str = " · ".join(g.title() for g in genre_cols if row[g] == 1)[:80] or "—"
                    score_color = "#E50914" if row['score'] >= 80 else ("#f5a623" if row['score'] >= 50 else "#888")
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div>
                                <div class="rec-title">{row['title']}</div>
                                <div class="rec-meta">
                                    {'Movie' if row['is_movie'] == 1 else 'TV Show'} &nbsp;·&nbsp;
                                    {int(row['release_year'])} &nbsp;·&nbsp;
                                    {row['rating']} &nbsp;·&nbsp;
                                    {genres_str}
                                </div>
                            </div>
                            <div class="rec-badge" style="background:{score_color};">{row['score']}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Similarity chart
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Similarity Score</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, max(3, len(recs) * 0.45)))
                bar_colors = ["#E50914" if s >= 80 else "#f5a623" if s >= 50 else "#555"
                              for s in recs['score']]
                ax.barh(recs['title'].str[:45][::-1], recs['score'][::-1], color=bar_colors[::-1])
                ax.set_xlabel("Similarity (%)")
                ax.set_xlim(0, 115)
                for i, (t, v) in enumerate(zip(recs['title'].str[:45][::-1], recs['score'][::-1])):
                    ax.text(v + 1, i, f"{v}%", va='center', fontsize=8, color="#888")
                plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    else:
        st.markdown(
            """
            <div style="text-align:center;padding:5rem 2rem;color:#444;">
                <div style="font-size:3.5rem;">🎬</div>
                <div style="font-size:0.95rem;margin-top:0.8rem;color:#555;">
                    Select a title above and click <strong style="color:#E50914;">Recommend →</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )