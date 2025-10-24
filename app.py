import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page configuration ---
st.set_page_config(
    page_title="GreenHouse Gases Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
view = st.sidebar.radio(
    "Select a View",
    ["Map View", "Trend Comparison", "Model Results", "CATE Visualization"]
)

# ------------------------------------------------------------------------
# MAP VIEW
# ------------------------------------------------------------------------
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

# ------------------------------------------------------------------------
# TREND COMPARISON
# ------------------------------------------------------------------------
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

# ------------------------------------------------------------------------
# MODEL RESULTS
# ------------------------------------------------------------------------
elif view == "Model Results":
    st.title("🧠 Causal Machine Learning Results")

    # --- Display model comparison table ---
    st.dataframe(causal_results, use_container_width=True)

    # --- Comparison plot across models ---
    if all(col in causal_results.columns for col in ["Model", "ATE", "CI_low", "CI_high"]):
        fig_comp = px.bar(
            causal_results,
            x="Model",
            y="ATE",
            color="Model",
            error_y=causal_results["CI_high"] - causal_results["ATE"],
            error_y_minus=causal_results["ATE"] - causal_results["CI_low"],
            title="Model Comparison: Average Treatment Effects with 95% Confidence Intervals"
        )
        fig_comp.update_layout(template="plotly_white")
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Confidence intervals not available for visualization.")

    # --- Identify and display the best model ---
    best_idx = causal_results["abs_ATE"].idxmax()
    best_model = causal_results.loc[best_idx, "Model"]
    ate = causal_results.loc[best_idx, "ATE"]
    ci_low = causal_results.loc[best_idx, "CI_low"]
    ci_high = causal_results.loc[best_idx, "CI_high"]

    st.metric(
        f"Best Model: {best_model}",
        f"ATE = {ate:.4f}",
        f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]"
    )

    

    # --- Add the essay interpretation ---
    st.markdown("### Interpretation and Policy Insights")
    st.markdown(
        """
        The causal analysis examined how economic growth, measured by GDP per capita, influences greenhouse gas (GHG)
        emissions across African economies after controlling for governance quality. Two models were estimated:
        a **Linear Double Machine Learning (LinearDML)** and a **Causal Forest Double Machine Learning (CausalForestDML)**.

        Both models found small, positive but statistically insignificant effects. For the LinearDML model,
        the average treatment effect (ATE) was **0.00063**, with a 95% confidence interval from **–0.0052 to 0.0065**.
        The CausalForestDML model yielded an ATE of **0.00336**, with a 95% confidence interval from **–0.0109 to 0.0176**.
        Because both intervals include zero, there is no statistically significant evidence that GDP per capita
        causally increases GHG emissions once governance quality is taken into account.

        This result suggests that the growth–environment relationship in Africa is not straightforward.
        Economic expansion, by itself, does not necessarily lead to higher emissions, especially in countries
        with effective governance structures. Good institutions can mediate the environmental impact of growth
        by enforcing environmental laws, promoting clean energy adoption, and supporting sustainable production practices.

        **Policy implications:** The findings highlight that economic and environmental goals need not conflict.
        Policymakers should focus on strengthening governance and institutional frameworks to ensure that economic
        progress is achieved sustainably. By investing in renewable energy, regulatory enforcement, and resource transparency,
        African nations can pursue growth that is environmentally responsible.

        The wide confidence intervals also indicate heterogeneity—meaning that some countries may experience stronger
        or weaker effects depending on their institutional context. Future work using **Conditional Average Treatment
        Effects (CATEs)** can uncover where growth most strongly drives emissions and where green transitions are working best.
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<style> p, li { text-align: justify; line-height: 1.6; } </style>",
        unsafe_allow_html=True
    )
st.info("Developed by: Christopher Mbiba ")
# ------------------------------------------------------------------------
# CATE VISUALIZATION
# ------------------------------------------------------------------------
elif view == "CATE Visualization":
    st.title("🎯 CATE Visualization (Best Model)")

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
