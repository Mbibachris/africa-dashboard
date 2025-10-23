import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Africa Dashboard", layout="wide")

# --- Load Data ---
df = pd.read_excel("data.xlsx")
causal_results = pd.read_excel("causal_results.xlsx")

# --- Dashboard ---
st.sidebar.header("Dashboard Controls")
view = st.sidebar.radio("Select View", ["Map", "Trends", "Model Results"])

if view == "Map":
    year = st.slider("Select Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].min()))
    variable = st.selectbox("Select Variable", ["ghg_emissions", "gdp_per_capita", "gov_effectiveness"])
    df_year = df[df["year"] == year]
    fig = px.choropleth(df_year,
                        locations="country",
                        locationmode="country names",
                        color=variable,
                        color_continuous_scale="YlOrRd",
                        title=f"{variable.replace('_', ' ').title()} in Africa ({year})")
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

elif view == "Trends":
    country = st.selectbox("Select Country", sorted(df["country"].unique()))
    variables = st.multiselect("Select Variables", ["ghg_emissions", "gdp_per_capita", "gov_effectiveness"],
                               default=["gdp_per_capita"])
    df_country = df[df["country"] == country]
    fig = px.line(df_country, x="year", y=variables,
                  title=f"{', '.join(variables).title()} over Time in {country}", markers=True)
    st.plotly_chart(fig, use_container_width=True)

elif view == "Model Results":
    st.subheader("Causal Machine Learning Results")
    st.dataframe(causal_results)
    ate = causal_results["ATE"].iloc[0]
    ci_low = causal_results["CI_low"].iloc[0]
    ci_high = causal_results["CI_high"].iloc[0]

    st.metric("Average Treatment Effect (ATE)", f"{ate:.4f}",
              f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    st.info("These results were estimated locally using EconML and imported into Streamlit.")
