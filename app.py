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
# Basic cleaning

df = df.copy()
df_plz = df_plz.copy()

# make plz_2 match in both dataframes
df["plz_2"] = df["plz_2"].astype(str).str.zfill(2)
df_plz["plz_2"] = df_plz["plz_2"].astype(str).str.zfill(2)

# rent per square meter
df["rent_sqr_m"] = df["baseRent"] / df["livingSpace"]

# building age group
df["building_age_group"] = np.where(df["yearConstructed"] >= 2010, "New", "Old")

# remove unrealistic values for cleaner plots
df = df[(df["rent_sqr_m"] > 0) & (df["rent_sqr_m"] < 50)]

# -----------------------------------------------------------------------------
# Region summary

offers_by_region = (
    df.groupby("plz_2")
    .size()
    .reset_index(name="offer_count")
)

rent_by_region = (
    df.groupby("plz_2")["rent_sqr_m"]
    .mean()
    .reset_index(name="avg_rent_sqr_m")
)

region_summary = offers_by_region.merge(rent_by_region, on="plz_2", how="left")

# use one row per region from df_plz
region_info = (
    df_plz[["plz_2", "region_name", "population"]]
    .drop_duplicates(subset="plz_2")
)

region_summary = region_summary.merge(region_info, on="plz_2", how="left")

region_summary["offers_per_1000"] = (
    region_summary["offer_count"] / region_summary["population"] * 1000
)

region_summary["investment_score"] = (
    region_summary["avg_rent_sqr_m"] / region_summary["offers_per_1000"]
)

region_summary = region_summary.replace([np.inf, -np.inf], np.nan)
region_summary = region_summary.dropna(
    subset=["avg_rent_sqr_m", "offers_per_1000", "investment_score"]
)



# -----------------------------------------------------------------------------
# Feature impact

feature_cols = ["balcony", "lift", "garden", "cellar", "hasKitchen", "newlyConst"]

feature_rows = []

for col in feature_cols:
    temp = df[[col, "rent_sqr_m"]].dropna()

    avg_yes = temp[temp[col] == True]["rent_sqr_m"].mean()
    avg_no = temp[temp[col] == False]["rent_sqr_m"].mean()

    feature_rows.append({
        "feature": col,
        "impact_on_rent_sqr_m": avg_yes - avg_no
    })

feature_impact = pd.DataFrame(feature_rows)
feature_impact = feature_impact.sort_values("impact_on_rent_sqr_m", ascending=True)

# -----------------------------------------------------------------------------
# Map data
# dissolve df_plz to one geometry per region

gdf_regions = df_plz.dissolve(by="plz_2", aggfunc="first").reset_index()

gdf_regions = gdf_regions.merge(
    region_summary[["plz_2", "offer_count", "avg_rent_sqr_m", "offers_per_1000", "investment_score"]],
    on="plz_2",
    how="left"
)

# -----------------------------------------------------------------------------
# Centered layout

left, center, right = st.columns([1, 2, 1])

# -----------------------------------------------------------------------------
# Title

with center:
    st.title("German  _housing_!")

    """
    This dashboard explores the German housing market from the perspective of a
    residential developer.

    The goal is to answer four questions: how the market is structured, where it
    may be worth building, what kind of apartments may create more value, and
    which regions appear most attractive overall.
    """

# -----------------------------------------------------------------------------
# Part I

with center:
    st.markdown("## Part I: Market structure")

    """
    **How does a selected city compare to the national rent distribution, and
    which regions appear relatively undersupplied?**
    """

    st.subheader("Rent distribution by city compared to Germany")

    city_list = sorted(df["geo_krs"].dropna().astype(str).unique())
    
    default_city = "Berlin"
    
    if default_city in city_list:
        default_index = city_list.index(default_city)
    else:
        default_index = 0
    selected_city = st.selectbox("Choose a city", city_list, index=default_index)

    df_city = df[df["geo_krs"].astype(str) == selected_city]

    germany_median = df["rent_sqr_m"].median()
    city_median = df_city["rent_sqr_m"].median()

    fig_hist = go.Figure()

    # Germany in the background
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

    # Selected city on top
    fig_hist.add_trace(
        go.Histogram(
            x=df_city["rent_sqr_m"],
            nbinsx=40,
            name=selected_city,
            opacity=0.75,
            histnorm="probability density",
            marker_color="cornflowerblue"
        )
    )

    # Germany median line
    fig_hist.add_vline(
        x=germany_median,
        line_color="lightgray",
        line_width=4,
        line_dash="dash"
    )

    # City median line
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
        title=f"Rent distribution in {selected_city} compared to Germany",
        xaxis_title="Rent per square meter",
        yaxis_title=None,
        xaxis=dict(range=[0, 35]),
        yaxis=dict(visible=False)
    )

    st.plotly_chart(fig_hist, use_container_width=True)
    
    """
    The full German market stays in the background, while the selected city is
    shown on top. This makes every city directly comparable to the national
    distribution.
    """

    st.subheader("Offers relative to population")

    geojson_data = gdf_regions.__geo_interface__

    fig_map = px.choropleth_mapbox(
        gdf_regions,
        geojson=geojson_data,
        locations="plz_2",
        featureidkey="properties.plz_2",
        color="offers_per_1000",
        mapbox_style="carto-positron",
        zoom=4.3,
        center={"lat": 51.2, "lon": 10.4},
        opacity=0.75,
        title="Offers per 1000 people by region"
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
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------------------------------------------------------
# Part II


with center:
    st.markdown("## Part II: Where to build?")

    """
    **Which regions reward modern buildings, and which regions combine low
    supply with high rent?**
    """

    st.subheader("Where do newer buildings command the strongest premium?")

    # -----------------------------------------------------------------------------
    # New vs old premium by region

    new_old_summary = (
        df.groupby(["plz_2", "building_age_group"])["rent_sqr_m"]
        .mean()
        .reset_index()
    )

    new_old_pivot = new_old_summary.pivot(
        index="plz_2",
        columns="building_age_group",
        values="rent_sqr_m"
    ).reset_index()

    new_old_pivot.columns.name = None

    if "New" not in new_old_pivot.columns:
        new_old_pivot["New"] = np.nan
    if "Old" not in new_old_pivot.columns:
        new_old_pivot["Old"] = np.nan

    new_old_pivot["premium_new_minus_old"] = new_old_pivot["New"] - new_old_pivot["Old"]

    new_old_pivot = new_old_pivot.merge(region_info, on="plz_2", how="left")
    new_old_pivot = new_old_pivot.dropna(subset=["New", "Old", "premium_new_minus_old"])

    # regions where old is more expensive
    bottom = new_old_pivot[new_old_pivot["premium_new_minus_old"] < 0].copy()

    # regions where new is more expensive
    top_6 = (
        new_old_pivot[new_old_pivot["premium_new_minus_old"] > 0]
        .sort_values("premium_new_minus_old", ascending=False)
        .head(6)
        .copy()
    )

    # combine them
    premium_display = pd.concat([bottom, top_6], ignore_index=True)

    # shorter labels
    premium_display["region_label"] = premium_display["region_name"].str.replace(" area", "", regex=False)

    # sort by difference so "old-valued" are together and "new-valued" are together
    premium_display = premium_display.sort_values("premium_new_minus_old", ascending=True)

    # where to place the difference label
    premium_display["label_x"] = premium_display[["New", "Old"]].max(axis=1) + 0.4

    # -----------------------------------------------------------------------------
    # Create overlay bar chart

    fig_premium = go.Figure()

    # Old bars in background
    fig_premium.add_trace(
        go.Bar(
            x=premium_display["Old"],
            y=premium_display["region_label"],
            orientation="h",
            name="Old buildings",
            marker_color="lightgray",
            opacity=0.95,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Old buildings: %{x:.2f} €/m²<extra></extra>"
            )
        )
    )

    # New bars on top
    fig_premium.add_trace(
        go.Bar(
            x=premium_display["New"],
            y=premium_display["region_label"],
            orientation="h",
            name="New buildings",
            marker_color="royalblue",
            opacity=0.85,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "New buildings: %{x:.2f} €/m²<extra></extra>"
            )
        )
    )

    # Difference labels
    fig_premium.add_trace(
        go.Scatter(
            x=premium_display["label_x"],
            y=premium_display["region_label"],
            mode="text",
            text=[f"{v:+.2f}" for v in premium_display["premium_new_minus_old"]],
            textfont=dict(color="dimgray", size=12),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    fig_premium.update_layout(
        barmode="overlay",
        title="New vs old building rents in regions with the strongest differences",
        plot_bgcolor="white",
        xaxis_title=None,
        yaxis_title=None,
        showlegend=True
    )

    fig_premium.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[0, premium_display["label_x"].max() + 1]
    )

    fig_premium.update_yaxes(
        showgrid=False
    )

    st.plotly_chart(fig_premium, use_container_width=True)


# -----------------------------------------------------------------------------

    st.subheader("Supply pressure vs rent")

    # -----------------------------------------------------------------------------
    # Calculate median split

    x_median = region_summary["offers_per_1000"].median()
    y_median = region_summary["avg_rent_sqr_m"].median()

    # -----------------------------------------------------------------------------
    # Create figure

    fig_scatter = go.Figure()

    fig_scatter.add_trace(
        go.Scatter(
            x=region_summary["offers_per_1000"],
            y=region_summary["avg_rent_sqr_m"],
            mode="markers",
            marker=dict(
                color="royalblue",
                size=9,
                opacity=0.75
            ),
            customdata=region_summary[["region_name"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Offers per 1000 people: %{x:.2f}<br>"
                "Average rent: %{y:.2f} €/m²<extra></extra>"
            ),
            showlegend=False
        )
    )

    # -----------------------------------------------------------------------------
    # Quadrant divider lines

    fig_scatter.add_vline(
        x=x_median,
        line_width=1.5,
        line_dash="dash",
        line_color="gray"
    )

    fig_scatter.add_hline(
        y=y_median,
        line_width=1.5,
        line_dash="dash",
        line_color="gray"
    )

    # -----------------------------------------------------------------------------
    # Quadrant labels

    x_min = region_summary["offers_per_1000"].min()
    x_max = region_summary["offers_per_1000"].max()
    y_min = region_summary["avg_rent_sqr_m"].min()
    y_max = region_summary["avg_rent_sqr_m"].max()

    fig_scatter.add_annotation(
        x=(x_min + x_median) / 2,
        y=(y_median + y_max) / 2,
        text="Low supply<br>High rent",
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    fig_scatter.add_annotation(
        x=(x_median + x_max) / 2,
        y=(y_median + y_max) / 2,
        text="High supply<br>High rent",
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    fig_scatter.add_annotation(
        x=(x_min + x_median) / 2,
        y=(y_min + y_median) / 2,
        text="Low supply<br>Low rent",
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    fig_scatter.add_annotation(
        x=(x_median + x_max) / 2,
        y=(y_min + y_median) / 2,
        text="High supply<br>Low rent",
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    # -----------------------------------------------------------------------------
    # Axis labels for "low" and "high" feeling

    fig_scatter.update_layout(
        title="Regional housing markets by supply pressure and rent level",
        plot_bgcolor="white",
        xaxis=dict(
            title="Supply (offers per 1000 people)",
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Rent level (€/m²)",
            showgrid=False,
            zeroline=False
        )
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# Part III

with center:
    st.markdown("## Part III: What to build?")

    """
    **Which features are associated with higher rents, and what apartment size
    appears most price-efficient for new buildings?**
    """

    st.subheader("Feature impact on rent")

    fig_features = px.bar(
        feature_impact,
        x="impact_on_rent_sqr_m",
        y="feature",
        orientation="h",
        text="impact_on_rent_sqr_m",
        title="Estimated feature premium on rent per square meter",
        labels={
            "impact_on_rent_sqr_m": "Average rent premium (€/m²)",
            "feature": "Feature"
        }
    )

    fig_features.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_features, use_container_width=True)

# ---------------------

  

    st.subheader("Apartment size vs rent per square meter for newer buildings")

    # -----------------------------------------------------------------------------
    # Prepare data

    df_new = df[df["building_age_group"] == "New"].copy()

    # add region_name into df_new using plz_2
    df_new = df_new.merge(
        region_info[["plz_2", "region_name"]],
        on="plz_2",
        how="left"
    )

    df_new = df_new.dropna(subset=["livingSpace", "rent_sqr_m", "region_name"])

    # crop outliers for cleaner view
    df_new = df_new[
        (df_new["livingSpace"] <= 250) &
        (df_new["rent_sqr_m"] <= 40)
    ].copy()

    # -----------------------------------------------------------------------------
    # Region selector

    region_list = sorted(df_new["region_name"].dropna().unique())

    default_region = "Berlin area"
    default_index = region_list.index(default_region) if default_region in region_list else 0

    selected_region = st.selectbox(
        "Highlight a region",
        region_list,
        index=default_index
    )

    df_region = df_new[df_new["region_name"] == selected_region].copy()

    # -----------------------------------------------------------------------------
    # Optional sampling for background points

    df_sample = df_new.sample(n=5000, random_state=42) if len(df_new) > 5000 else df_new

    # -----------------------------------------------------------------------------
    # Build trend line from binned averages

    df_new["size_bin"] = pd.cut(df_new["livingSpace"], bins=30)

    trend = (
        df_new.groupby("size_bin", observed=False)["rent_sqr_m"]
        .mean()
        .reset_index()
    )

    trend["size_mid"] = trend["size_bin"].apply(lambda x: x.mid)

    # -----------------------------------------------------------------------------
    # Create figure

    fig_size = go.Figure()

    # All regions in background
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

    # Selected region highlighted
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

    # Overall trend line
    fig_size.add_trace(
        go.Scatter(
            x=trend["size_mid"],
            y=trend["rent_sqr_m"],
            mode="lines",
            line=dict(color="black", width=3),
            name="Overall trend"
        )
    )

    # -----------------------------------------------------------------------------
    # Layout

    fig_size.update_layout(
        title="How rent per m² changes with apartment size in newer buildings",
        xaxis_title="Apartment size (m²)",
        yaxis_title="Rent per m²",
        plot_bgcolor="white"
    )

    fig_size.update_xaxes(
        range=[0, 250],
        showgrid=False
    )

    fig_size.update_yaxes(
        range=[0, 40],
        showgrid=False
    )

    st.plotly_chart(fig_size, use_container_width=True)

# -----------------------------------------------------------------------------
# Part IV

with center:
    st.markdown("## Part IV: Final investment score")

    """
    **Which regions combine high rent with relatively low supply?**
    """

    top_10_score = region_summary.sort_values("investment_score", ascending=False).head(10)

    fig_score = px.bar(
        top_10_score.sort_values("investment_score", ascending=True),
        x="investment_score",
        y="region_name",
        orientation="h",
        text="investment_score",
        title="Top 10 regions by investment score",
        labels={
            "investment_score": "Investment score",
            "region_name": "Region"
        }
    )

    fig_score.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_score, use_container_width=True)

    st.subheader("Top 10 regions table")

    st.dataframe(
        top_10_score[
            ["region_name", "offer_count", "population", "offers_per_1000", "avg_rent_sqr_m", "investment_score"]
        ].reset_index(drop=True)
    )