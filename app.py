import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Africa Energy & Development Dashboard", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_excel("data.xlsx")
    causal = pd.read_excel("causal_results (1).xlsx")
    cate = pd.read_excel("cate_results.xlsx")
    return df, causal, cate

df, causal_results, cate_results = load_data()

# --- Sidebar ---
st.sidebar.title("Navigation")
view = st.sidebar.radio("Select a View", [
    "Map View",
    "Trend Comparison",
    "Model Comparison",
    "CATE Visualization"
])

# --- MAP VIEW ---
if view == "Map View":
    st.title("🌍 Africa Map Visualization")

    year = st.slider(
        "Select Year",
        int(df["year"].min()), int(df["year"].max()),
        int(df["year"].min())
    )

    variable = st.selectbox(
        "Select Variable to Display",
        ["ghg_emissions", "gdp_per_capita", "gov_effectiveness"]
    )

    df_year = df[df["year"] == year]

    fig = px.choropleth(
        df_year,
        locations="country",
        locationmode="country names",
        color=variable,
        color_continuous_scale="YlOrRd",
        title=f"{variable.replace('_', ' ').title()} across Africa ({year})"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- TREND COMPARISON ---
elif view == "Trend Comparison":
    st.title("📈 Country Trend Comparison")

    countries = st.multiselect(
        "Select Countries to Compare",
        sorted(df["country"].unique()),
        default=["Ghana", "Nigeria", "South Africa"][:3]
    )

    variable = st.selectbox(
        "Select Variable",
        ["ghg_emissions", "gdp_per_capita", "gov_effectiveness"]
    )

    if countries:
        df_filtered = df[df["country"].isin(countries)]
        fig = px.line(
            df_filtered,
            x="year",
            y=variable,
            color="country",
            markers=True,
            title=f"Trends in {variable.replace('_', ' ').title()} (Selected Countries)"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one country to display trends.")

# --- MODEL COMPARISON ---
elif view == "Model Comparison":
    st.title("⚖️ Causal Model Comparison: ATE Estimates")

    # Show table
    st.dataframe(causal_results, use_container_width=True)

    # Bar chart of ATE with CI
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=causal_results["Model"],
        y=causal_results["ATE"],
        error_y=dict(
            type='data',
            array=(causal_results["CI_high"] - causal_results["ATE"]).abs(),
            visible=True
        ),
        name="ATE",
        marker_color="indianred"
    ))
    fig.update_layout(
        title="Average Treatment Effect (ATE) with 95% Confidence Intervals",
        yaxis_title="ATE Estimate",
        xaxis_title="Model",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- CATE VISUALIZATION ---
elif view == "CATE Visualization":
    st.title("🎯 CATE Visualization (Best Model)")

    # Summary stats
    st.markdown(f"**Number of CATE observations:** {len(cate_results):,}")
    st.markdown(f"**Mean CATE:** {cate_results['CATE'].mean():.4f}")
    st.markdown(f"**Std Dev:** {cate_results['CATE'].std():.4f}")

    col1, col2 = st.columns(2)

    with col1:
        # Distribution plot
        fig = px.histogram(
            cate_results,
            x="CATE",
            nbins=40,
            title="Distribution of Conditional Average Treatment Effects (CATE)",
            color_discrete_sequence=["royalblue"]
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scatter plot
        fig2 = px.scatter(
            cate_results,
            x="gdp_per_capita",
            y="CATE",
            color="gov_effectiveness",
            hover_data=["country"] if "country" in cate_results.columns else None,
            title="CATE vs GDP per Capita (Colored by Government Effectiveness)"
        )
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
