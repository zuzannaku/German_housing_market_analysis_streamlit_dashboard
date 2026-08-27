from pathlib import Path
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="German housing",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_DIR / "apartments_clean.parquet")
    df_plz = gpd.read_file(DATA_DIR / "plz_clean.geojson")
    return df, df_plz

df, df_plz = load_data()

# -----------------------------------------------------------------------------
# Tiny preprocessing

df["rent_sqr_m"] = df["baseRent"] / df["livingSpace"]
df = df[(df["rent_sqr_m"] > 0) & (df["rent_sqr_m"] < 50)]

df["building_age_group"] = np.where(df["yearConstructed"] >= 2010, "New", "Old")


# -----------------------------------------------------------------------------
# Tiny feature engineering
# Regional summary including the number of listing per region, rent/sqr m per region, population per region, offers/1000 people per region
# and the new investment score im computing

region_info = (
    df_plz[["plz_2", "region_name", "population"]]
    .drop_duplicates(subset="plz_2")
)

region_summary = (
    df.groupby("plz_2")
    .agg(
        offer_count=("plz_2", "size"),
        avg_rent_sqr_m=("rent_sqr_m", "mean")
    )
    .reset_index()
    .merge(region_info, on="plz_2", how="left")
)

region_summary["offers_per_1000"] = region_summary["offer_count"] / region_summary["population"] * 1000
region_summary["investment_score"] = region_summary["avg_rent_sqr_m"] / region_summary["offers_per_1000"]

region_summary = region_summary.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["avg_rent_sqr_m", "offers_per_1000", "investment_score"]
)

# -----------------------------------------------------------------------------
# Preparing data for making the map

gdf_regions = df_plz.dissolve(by="plz_2", aggfunc="first").reset_index()

gdf_regions = gdf_regions.merge(
    region_summary[["plz_2", "offer_count", "avg_rent_sqr_m", "offers_per_1000", "investment_score"]],
    on="plz_2",
    how="left"
)

# -----------------------------------------------------------------------------
# Start of the app

left, center, right = st.columns([1, 3, 1])

with center:
    st.title("German  _housing market_ analysis")

    """
    This dashboard explores the German rental market to find areas with high potential for investments 
    and identify the features that influence rent prices. It considers factors such as regional supply, pricing and apartment features 
    to provide insights and support more informed, data-driven investment decisions.

    The analysis was structured around the following parts:
    - Regional based analysis of the rental market.
    - Identification of most promising areas.
    - Relationship between property features and rent prices.
    
    """

# -----------------------------------------------------------------------------
# Part I

with center:
    st.markdown("## Part I: Regional market overview")

    """
    The following section provides figures that help better understand the 
    overall rental market in Germany. This part is crucial for creating context for the 
    more complex figures in the following parts of the dashboard.
    """
# ---------------- GRAPH 1

    st.subheader("Rent distribution by city compared to Germany")

    city_list = sorted(
        df["geo_krs"]
        .dropna()
        .astype(str)
        .str.replace("_", " ", regex=False)
        .unique()
    )
    default_city = "Berlin"
    default_index = city_list.index(default_city)
    selected_city = st.selectbox("Choose a city", city_list, index=default_index)
    df_city = df[df["geo_krs"].astype(str).str.replace("_", " ", regex=False) == selected_city]

    germany_median = df["rent_sqr_m"].median()
    city_median = df_city["rent_sqr_m"].median()

    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Histogram(
            x=df["rent_sqr_m"],
            nbinsx=40,
            name="Germany",
            opacity=0.5,
            histnorm="probability density",
            marker_color="lightgray"
        )
    )

    fig_hist.add_trace(
        go.Histogram(
            x=df_city["rent_sqr_m"],
            nbinsx=40,
            name=selected_city,
            opacity=0.75,
            histnorm="probability density",
            marker_color="#2F80ED"
        )
    )

    fig_hist.add_vline(
        x=germany_median,
        line_color="lightgray",
        line_width=4,
        line_dash="dash"
    )

    fig_hist.add_vline(
        x=city_median,
        line_color="royalblue",
        line_width=4,
        line_dash="dash"
    )
    
    fig_hist.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="Median"
        )
    )

    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Rent level (€/m²)",
        yaxis_title=None,
        xaxis=dict(range=[0, 35]),
        yaxis=dict(visible=False),
        margin=dict(t=10),
        legend=dict(
            font=dict(size=14)
        )
    )

    st.plotly_chart(fig_hist, use_container_width=True)
    
    
    """
    Regions for which the distribution is shifted to the right and the median is higher that for Germany 
    could be potential markets with higher returns for investors.
    """
# ---------------- GRAPH 2

    st.subheader("Offers relative to population")

    geojson_data = gdf_regions.__geo_interface__

    fig_map = px.choropleth_map(
        gdf_regions,
        geojson=geojson_data,
        locations="plz_2",
        featureidkey="properties.plz_2",
        color="offers_per_1000",
        labels={"offers_per_1000": "Offers per <b>1000 residents"},
        color_continuous_scale="dense",
        map_style="carto-positron",
        zoom=4.3,
        center={"lat": 51.2, "lon": 10.4},
        opacity=0.8
    )

    fig_map.update_traces(
        customdata=gdf_regions[["region_name", "offers_per_1000"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Offer density: %{customdata[1]:.2f}"
            "<extra></extra>"
        )
    )

    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    st.plotly_chart(fig_map, use_container_width=True)

    """
    Regions with higher offer density may suggest a more competitive local market, 
    while those with low offer density could signify worse availability of offers, 
    which then could lead to overstated rent prices.
    """
# -----------------------------------------------------------------------------
# Part II

with center:
    st.markdown("## Part II: Where to build?")

    """
    This section help further identify what regions have high investment potential. 
    It explores areas with preference for newly constructed buildings and those that 
    combine low supply of available offers and high rent prices. Those regions are most 
    likely to be a "seller's market" and are therefore, the ideal location for a new investment.
    
    """
# ---------------- GRAPH 3

    st.subheader("What regions have the biggest rent gap based on building age?")

    df_grouped = (
        df.groupby(["plz_2", "building_age_group"])["rent_sqr_m"]
        .mean()
        .unstack()
        .reset_index()
    )

    df_grouped["premium"] = df_grouped["New"] - df_grouped["Old"]
    df_grouped = df_grouped.merge(region_info, on="plz_2", how="left")

    df_display = (
        df_grouped.sort_values("premium", ascending=False)
        .head(8)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_display["region_name"],
            y=df_display["New"],
            name="New buildings <br>(newer that 2010)",
            marker_color="#2F80ED",
            opacity=0.8
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=df_display["region_name"],
            y=df_display["Old"],
            name="Old buildings <br>(older than 2010)",
            marker_color="lightgray",
            opacity=1
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_display["region_name"],
            y=df_display[["New", "Old"]].max(axis=1) + 0.5,
            mode="text",
            text=[f"{v:+.2f}" for v in df_display["premium"]],
            textfont=dict(color="black", size=14),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        barmode="overlay",
        plot_bgcolor="white",
        xaxis_title=None,
        yaxis_title=None,
        xaxis_tickangle=-30,
        margin=dict(t=10),
        legend=dict(
            font=dict(size=14)
        )
    )
    
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False
    )
    fig.update_xaxes(
        showgrid=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(""" 
    Newly constructed buildings are valued amongst all regions.
    Presented on the visualisation markets have the highest difference in rents between old and new building,
    suggesting higher potential returns for new properties.    
    """)

# ---------------- GRAPH 4

    st.subheader("Supply pressure vs rent")

    x_median = region_summary["offers_per_1000"].median()
    y_median = region_summary["avg_rent_sqr_m"].median()

    fig_scatter = go.Figure()

    fig_scatter.add_trace(
        go.Scatter(
            x=region_summary["offers_per_1000"],
            y=region_summary["avg_rent_sqr_m"],
            mode="markers",
            marker=dict(
                color="#2F80ED",
                size=10,
                opacity=0.8
            ),
            customdata=region_summary[["region_name"]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Offers per 1000 residents: %{x:.2f}<br>"
                "Average rent: %{y:.2f} €/m²<extra></extra>"
            ),
            showlegend=False
        )
    )

    fig_scatter.add_vline(
        x=x_median,
        line_width=2,
        line_dash="dash",
        line_color="lightgray"
    )

    fig_scatter.add_hline(
        y=y_median,
        line_width=2,
        line_dash="dash",
        line_color="lightgray"
    )

    fig_scatter.add_shape(
        type="rect",
        x0=-15,
        x1=x_median,
        y0=y_median,
        y1=27,
        fillcolor="aliceblue",
        line=dict(width=0),
        layer="below"
    )

    fig_scatter.add_annotation(
        x=0.02,
        y=0.95,
        xref="paper",
        yref="paper",
        text="Low supply<br>High rent",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="left"
    )

    fig_scatter.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        text="High supply<br>High rent",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right"
    )

    fig_scatter.add_annotation(
        x=0.02,
        y=0.05,
        xref="paper",
        yref="paper",
        text="Low supply<br>Low rent",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="left"
    )

    fig_scatter.add_annotation(
        x=0.98,
        y=0.05,
        xref="paper",
        yref="paper",
        text="High supply<br>Low rent",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right"
    )
    
    fig_scatter.update_yaxes(
        range=[-3, 27]
    )
    fig_scatter.update_xaxes(
        range=[-15, 310]
    )

    fig_scatter.update_layout(
        plot_bgcolor="white",
        xaxis=dict(
            title="Supply (offers per 1000 residents)",
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Rent level (€/m²)",
            showgrid=False,
            zeroline=False
        ),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("""
    Regions in the highlighted, top-left quadrant combine high rents with relatively low supply, signifying a seller's market, 
    where demand is higher that supply. 
    This makes them the most attractive areas for new investments. On the other hand regions with high supply and 
    lower rents may indicate more saturated markets in the country.
    """)

# -----------------------------------------------------------------------------
# Part III

with center:
    st.markdown("## Part III: What to build?")

    """
    After identifying key markets for investments, another important insight is to find what kind of features drive rents higher, and what apartment size
    is the most price-efficient.
    """

# ---------------- GRAPH 5

df_new = df[df["yearConstructed"] >= 2010]
feature_cols = ["balcony", "lift", "garden", "cellar", "hasKitchen"]

feature_rows = []

for col in feature_cols:
    temp = df_new[[col, "rent_sqr_m"]].dropna()

    avg_yes = temp[temp[col] == True]["rent_sqr_m"].mean()
    avg_no = temp[temp[col] == False]["rent_sqr_m"].mean()

    feature_rows.append({
        "feature": col,
        "impact_on_rent_sqr_m": avg_yes - avg_no
    })

feature_impact = pd.DataFrame(feature_rows)
feature_impact["feature"] = feature_impact["feature"].replace({
    "balcony": "Balcony",
    "lift": "Lift",
    "garden": "Garden",
    "cellar": "Cellar",
    "hasKitchen": "Kitchen"
})

feature_impact = feature_impact.sort_values("impact_on_rent_sqr_m", ascending=True)

feature_impact["label"] = feature_impact["impact_on_rent_sqr_m"].apply(
    lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}"
)

colors = [
    "#2F80ED" if v > 0 else "#B0B0B0"
    for v in feature_impact["impact_on_rent_sqr_m"]
]

with center:
    st.subheader("Feature impact on rent (€/m²) for buildings newer than 2010")
    
    fig_features = px.bar(
        feature_impact,
        x="impact_on_rent_sqr_m",
        y="feature",
        orientation="h",
        text="label",
        opacity=0.8
    )
    fig_features.update_traces(marker_color=colors)
    
    fig_features.update_traces(
        textposition="outside",
        textfont=dict(size=14, color="black")
    )

    fig_features.update_xaxes(
        visible=False
    )

    fig_features.update_layout(
        yaxis_title=None,
        plot_bgcolor="white",
        margin=dict(t=10, b=10)
    )

    st.plotly_chart(fig_features, use_container_width=True)

    """
    The results of this visualisation show what features are the most valuable for customers. 
    In the proccess of planing the property development, in order to maximalize the profit, it is important to consider exacly these features.
    """

# ---------------- GRAPH 6

    st.subheader("Apartment size vs rent for buildings newer than 2010")
    n_points = len(df_new)
    sample_size = min(n_points, 5000)
    df_sample = df_new.sample(n=sample_size, random_state=42)
    st.caption(f"Showing {sample_size:,} out of {n_points:,} data points")

    df_new = df[df["building_age_group"] == "New"].copy()
    df_new = df_new.merge(
        region_info[["plz_2", "region_name"]],
        on="plz_2",
        how="left"
    )

    region_list = sorted(df_new["region_name"].dropna().unique())
    default_region = "Berlin area"
    default_index = region_list.index(default_region) if default_region in region_list else 0
    selected_region = st.selectbox(
        "Highlight a region",
        region_list,
        index=default_index
    )

    df_region = df_new[df_new["region_name"] == selected_region].copy()

    df_new["size_bin"] = pd.cut(df_new["livingSpace"], bins=30)

    trend = (
        df_new.groupby("size_bin", observed=False)["rent_sqr_m"]
        .mean()
        .reset_index()
    )

    trend["size_mid"] = trend["size_bin"].apply(lambda x: x.mid)

    fig_size = go.Figure()

    fig_size.add_trace(
        go.Scatter(
            x=df_sample["livingSpace"],
            y=df_sample["rent_sqr_m"],
            mode="markers",
            marker=dict(
                color="lightgray",
                size=4,
                opacity=0.25
            ),
            name="All regions",
            hoverinfo="skip"
        )
    )

    fig_size.add_trace(
        go.Scatter(
            x=df_region["livingSpace"],
            y=df_region["rent_sqr_m"],
            mode="markers",
            marker=dict(
                color="royalblue",
                size=5,
                opacity=0.55
            ),
            name=selected_region,
            customdata=df_region[["region_name"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Apartment size: %{x:.0f} m²<br>"
                "Rent per m²: %{y:.2f} €<extra></extra>"
            )
        )
    )

    fig_size.add_trace(
        go.Scatter(
            x=trend["size_mid"],
            y=trend["rent_sqr_m"],
            mode="lines",
            line=dict(color="black", width=3),
            name="Overall trend"
        )
    )

    fig_size.update_layout(
        xaxis_title="Apartment size (m²)",
        yaxis_title="Rent level (€/m²)",
        plot_bgcolor="white",
        margin=dict(t=10)
    )

    fig_size.update_xaxes(
        range=[0, 250],
        showgrid=False
    )

    fig_size.update_yaxes(
        range=[0, 35],
        showgrid=False
    )

    st.plotly_chart(fig_size, use_container_width=True)

    """
    The overall trend of the rent level based on apartment size, presents a 
    very important insight regarding the optimal size of the properties.
    For smaller apartments we see a rapid decline until it reaches area 50m². Beyond this point, 
    the rent per square meter remains relatively constant, suggesting diminishing returns to size. 
    This means that appartment with their total area below 50m² could be the most price-efficient and bring in the most profit.
    """

# END OF NICE GRAPHS -------------------

# Part IV

with center:
    st.markdown("## Part IV: Final investment scores")

    """
    **Top 10 regions that combine high rent with low supply**
    """

    top_10_score = (
    region_summary
    .sort_values("investment_score", ascending=False)
    .head(10)
)

    top_10_score["label"] = top_10_score["investment_score"].apply(lambda x: f"{x:.2f}")

    fig_score = px.bar(
        top_10_score,
        x="investment_score",
        y="region_name",
        orientation="h",
        text="label",
        opacity=0.8
    )

    fig_score.update_traces(
        marker_color="#2F80ED",
        textposition="outside",
        textfont=dict(size=14, color="black")
    )

    fig_score.update_xaxes(
        visible=False
    )
    
    fig_score.update_yaxes(
        autorange="reversed"
    )

    fig_score.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor="white",
        margin=dict(t=10, b=20)
    )

    st.plotly_chart(fig_score, use_container_width=True)

    st.subheader("Top 10 regions table")
    
    top_10_score_again = (
        region_summary
        .sort_values("investment_score", ascending=False)
        .head(10)
    )
    
    st.dataframe(
        top_10_score_again[
            ["region_name", "offer_count", "population", "offers_per_1000", "avg_rent_sqr_m", "investment_score"]
        ].reset_index(drop=True)
    )
    st.markdown("## Conclusion")
    
    """
    The investment score combines the information about regional offer count, regional population, rent price and apartment size.
    This way we indentified top 10 potential regions for future property development that should be the topic of future, more detailed analysis.
    Additoinally, we recognised what apartment features are the most relevant for customers. A lift increases the rent per square meter on average by 1.85€, 
    while a furnished kitchen by 2.36€. 
    The last insigh, regarding the optimal size of the apartment, suggest that total area below 50m² is the most price-efficient and could bring the best return on investment.

    """
    
