import streamlit as st
import pandas as pd
import plotly.express as px
from econml.dml import LinearDML, CausalForestDML
from econml.dr import DRLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(page_title="Africa Economic Dashboard", layout="wide")
st.title("🌍 Africa Economic Dashboard with Causal ML Insights")

# ==============================
# DATA UPLOAD / LOADING
# ==============================
uploaded_file = st.file_uploader("📂 Upload your cleaned CSV data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully!")
    st.dataframe(df.head())
else:
    st.warning("Upload your dataset to continue.")
    st.stop()

# ==============================
# CREATE TABS
# ==============================
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Causal Modeling"])

# ==============================
# TAB 1 — DASHBOARD
# ==============================
with tab1:
    st.subheader("Explore African Economic Indicators")

    countries = sorted(df["country"].unique())
    years = sorted(df["year"].unique())
    variables = [col for col in df.columns if col not in ["country", "year"]]

    view = st.radio("Select View", ["Map View", "Trend View", "Comparison View"])

    if view == "Map View":
        var = st.selectbox("Select Variable", variables)
        year = st.selectbox("Select Year", years)
        df_year = df[df["year"] == year]

        fig = px.choropleth(
            df_year,
            locations="country",
            locationmode="country names",
            color=var,
            color_continuous_scale="YlOrRd",
            title=f"{var.replace('_',' ').title()} in Africa ({year})",
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Trend View":
        country = st.selectbox("Select Country", countries)
        vars_selected = st.multiselect("Select Variables", variables, variables[:2])

        fig = px.line(
            df[df["country"] == country],
            x="year",
            y=vars_selected,
            title=f"{', '.join(vars_selected).title()} Over Time in {country}",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Comparison View":
        var = st.selectbox("Select Variable to Compare", variables)
        fig = px.line(
            df,
            x="year",
            y=var,
            color="country",
            title=f"{var.replace('_',' ').title()} Across African Countries",
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# TAB 2 — CAUSAL MODELING
# ==============================
with tab2:
    st.subheader("Causal Machine Learning (EconML)")

    outcome = st.selectbox("Select Outcome Variable (Y)", variables)
    treatment = st.selectbox("Select Treatment Variable (T)", variables)
    controls = st.multiselect(
        "Select Control Variables (X)",
        [v for v in variables if v not in [outcome, treatment]]
    )

    model_choice = st.radio(
        "Select Model Type",
        [
            "LinearDML (baseline)",
            "DRLearner (semi-parametric)",
            "CausalForestDML (heterogeneous effects)"
        ],
        horizontal=False
    )

    if st.button("Run Causal Estimation"):
        from econml.dml import LinearDML, CausalForestDML
        from econml.dr import DRLearner
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        df_clean = df[[outcome, treatment] + controls].dropna()
        Y = df_clean[outcome]
        T = df_clean[treatment]
        X = df_clean[controls] if controls else None

        if "LinearDML" in model_choice:
            est = LinearDML(
                model_y=RandomForestRegressor(),
                model_t=RandomForestRegressor(),
                linear_model=LinearRegression(),
                random_state=42,
            )
        elif "DRLearner" in model_choice:
            est = DRLearner(
                model_regression=RandomForestRegressor(),
                model_propensity=RandomForestRegressor(),
                random_state=42,
            )
        elif "CausalForestDML" in model_choice:
            est = CausalForestDML(
                model_t=RandomForestRegressor(),
                model_y=RandomForestRegressor(),
                random_state=42,
            )

        est.fit(Y, T, X=X)
        ate = est.ate(X)
        ci = est.ate_interval(X)

        st.success(f"✅ Model estimation complete using {model_choice}!")
        st.markdown(f"""
        ### Average Treatment Effect (ATE)
        **ATE:** {ate:.4f}  
        **95% Confidence Interval:** [{ci[0]:.4f}, {ci[1]:.4f}]
        """)

        if hasattr(est, "effect"):
            cate = est.effect(X)
            st.markdown("### Conditional Average Treatment Effects (CATE)")
            st.dataframe(pd.DataFrame(cate, columns=["Effect"]).head(10))


st.sidebar.info("Built by Christopher Mbiba")
