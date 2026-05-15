import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

# ── PAGE SETTINGS ──
st.set_page_config(
    page_title="Energy Consumption Analysis",
    page_icon="⚡",
    layout="wide"
)

# ── CUSTOM CSS ──
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    h1 { color: #00d4ff; text-align: center; font-size: 2.5rem; }
    h2 { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 8px; }
    h3 { color: #ffffff; }
    .stMetric { background-color: #1e2130; border-radius: 10px; padding: 10px; }
    .stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚡ World Energy Consumption Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaaaaa;'>Energy Analysis with Machine Learning and Artificial Intelligence</p>", unsafe_allow_html=True)
st.markdown("---")

# ── LOAD DATA ──
@st.cache_data
def load_data():
    df = pd.read_csv("data/World Energy Consumption.csv")
    columns = ['country', 'year', 'population', 'gdp',
               'primary_energy_consumption', 'fossil_fuel_consumption', 'renewables_consumption']
    return df[columns].dropna()

df = load_data()

# ── REGIONS ──
regions = {
    "🌍 Europe":      ["Germany", "France", "Turkey", "Italy", "Spain", "United Kingdom", "Poland"],
    "🌏 Asia":        ["China", "Japan", "India", "South Korea", "Indonesia"],
    "🌎 Americas":    ["United States", "Canada", "Brazil", "Mexico", "Argentina"],
    "🌎 Middle East": ["Saudi Arabia", "Iran", "Iraq", "United Arab Emirates"],
    "🌍 Africa":      ["South Africa", "Egypt", "Nigeria", "Algeria"]
}

# ══════════════════════════════════════════
# TABS
# ══════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Overview",
    "📊 Analysis & Heatmap",
    "🔮 Forecast",
    "🤖 AI Models"
])

# ══════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════
with tab1:
    st.header("Factors Affecting Energy Consumption")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("👥 **Population**\n\nAs population grows, energy demand rises directly. More people means more housing, transportation, and consumption.")
    with col2:
        st.info("💰 **Economic Growth (GDP)**\n\nCountries with growing economies produce more. Industry and trade increase energy demand.")
    with col3:
        st.info("🌡️ **Climate & Geography**\n\nCold countries spend more energy on heating, hot countries on cooling.")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.success("🏭 **Industrialization**\n\nHeavy industry and factories are the largest energy consumers in a country.")
    with col5:
        st.success("⚡ **Energy Efficiency**\n\nAs technology advances, it becomes possible to do the same work with less energy.")
    with col6:
        st.success("🌱 **Renewable Energy**\n\nSources like solar and wind reduce dependence on fossil fuels.")

    st.markdown("---")
    st.header("📋 About the Dataset")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        row_count = len(df)
        st.metric("Total Records", f"{row_count:,}")
    with col_b:
        country_count = df['country'].nunique()
        st.metric("Countries", str(country_count))
    with col_c:
        st.metric("Year Range", "1965 - 2018")
    with col_d:
        st.metric("Features", "7")

    st.markdown("---")
    st.subheader("Column Descriptions")

    dictionary = {
        "Column": ["country", "year", "population", "gdp",
                   "primary_energy_consumption", "fossil_fuel_consumption", "renewables_consumption"],
        "Description": [
            "Country name",
            "Year",
            "Population",
            "Economic size (dollars)",
            "Total energy consumption (TWh)",
            "Fossil fuel consumption (TWh)",
            "Renewable energy consumption (TWh)"
        ]
    }
    st.dataframe(pd.DataFrame(dictionary), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════
# TAB 2 — ANALYSIS & HEATMAP
# ══════════════════════════════════════════
with tab2:
    st.header("Country Comparison by Region")

    selected_region = st.selectbox("Select region:", list(regions.keys()))
    region_countries = [c for c in regions[selected_region] if c in df['country'].values]

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    fig1.patch.set_facecolor('#0e1117')
    ax1.set_facecolor('#1e2130')
    colors_region = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b', '#74c0fc']
    for country, color in zip(region_countries, colors_region):
        data = df[df['country'] == country]
        ax1.plot(data['year'], data['primary_energy_consumption'],
                 label=country, color=color, marker='o', markersize=3, linewidth=2)
    ax1.set_title(f'{selected_region} — Annual Energy Consumption', color='white')
    ax1.set_xlabel('Year', color='white')
    ax1.set_ylabel('Energy Consumption (TWh)', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1e2130', labelcolor='white')
    ax1.spines['bottom'].set_color('#444')
    ax1.spines['left'].set_color('#444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    st.pyplot(fig1)

    st.markdown("---")
    st.header("🌡️ Heatmap — Consumption Over Years")
    st.write("Each row is a country, each column is a year. Darker color means higher consumption.")

    heatmap_countries = region_countries
    years = sorted(df['year'].unique())

    # Build matrix
    matrix = []
    for country in heatmap_countries:
        row = []
        for year in years:
            value = df[(df['country'] == country) & (df['year'] == year)]['primary_energy_consumption']
            row.append(value.values[0] if len(value) > 0 else 0)
        matrix.append(row)

    matrix_np = np.array(matrix)

    fig2, ax2 = plt.subplots(figsize=(16, len(heatmap_countries) * 0.8 + 2))
    fig2.patch.set_facecolor('#0e1117')
    ax2.set_facecolor('#0e1117')

    im = ax2.imshow(matrix_np, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax2, label='Energy Consumption (TWh)')

    ax2.set_xticks(range(0, len(years), 5))
    ax2.set_xticklabels([years[i] for i in range(0, len(years), 5)], rotation=45, color='white')
    ax2.set_yticks(range(len(heatmap_countries)))
    ax2.set_yticklabels(heatmap_countries, color='white')
    ax2.set_title('Heatmap — Energy Consumption Over Years', color='white', pad=15)
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")
    st.header("🏆 Top 10 Countries by Consumption in 2018")
    last_year = df[df['year'] == 2018].sort_values('primary_energy_consumption', ascending=False).head(10)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    fig3.patch.set_facecolor('#0e1117')
    ax3.set_facecolor('#1e2130')
    ax3.bar(last_year['country'], last_year['primary_energy_consumption'],
            color='#00d4ff', edgecolor='#0099bb')
    ax3.set_xlabel('Country', color='white')
    ax3.set_ylabel('Energy Consumption (TWh)', color='white')
    ax3.tick_params(colors='white')
    plt.xticks(rotation=45)
    ax3.spines['bottom'].set_color('#444')
    ax3.spines['left'].set_color('#444')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

# ══════════════════════════════════════════
# TAB 3 — FORECAST
# ══════════════════════════════════════════
with tab3:
    st.header("Select a Country and Forecast Future Consumption")

    col_a, col_b = st.columns(2)
    with col_a:
        selected_region2 = st.selectbox("Select region:", list(regions.keys()), key="b2")
    with col_b:
        region_countries2 = [c for c in regions[selected_region2] if c in df['country'].values]
        selected_country2 = st.selectbox("Select country:", region_countries2, key="u2")

    country_df = df[df['country'] == selected_country2].copy()

    if len(country_df) >= 10:

        # Stat cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak Consumption",
                      f"{country_df['primary_energy_consumption'].max():.0f} TWh",
                      f"Year {int(country_df.loc[country_df['primary_energy_consumption'].idxmax(), 'year'])}")
        with col2:
            st.metric("Lowest Consumption",
                      f"{country_df['primary_energy_consumption'].min():.0f} TWh",
                      f"Year {int(country_df.loc[country_df['primary_energy_consumption'].idxmin(), 'year'])}")
        with col3:
            growth = country_df['primary_energy_consumption'].pct_change().mean() * 100
            st.metric("Avg. Annual Growth", f"{growth:.1f}%")

        st.markdown("---")

        # Fossil vs Renewable
        st.subheader(f"{selected_country2} — Fossil Fuel vs Renewable Energy")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        fig4.patch.set_facecolor('#0e1117')
        ax4.set_facecolor('#1e2130')
        ax4.plot(country_df['year'], country_df['fossil_fuel_consumption'],
                 label='Fossil Fuel', color='#ff6b6b', linewidth=2)
        ax4.plot(country_df['year'], country_df['renewables_consumption'],
                 label='Renewables', color='#51cf66', linewidth=2)
        ax4.set_xlabel('Year', color='white')
        ax4.set_ylabel('Energy Consumption (TWh)', color='white')
        ax4.tick_params(colors='white')
        ax4.legend(facecolor='#1e2130', labelcolor='white')
        ax4.spines['bottom'].set_color('#444')
        ax4.spines['left'].set_color('#444')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        st.pyplot(fig4)

        st.markdown("---")

        # Future forecast
        st.subheader("📅 Forecast to 2030")
        poly       = PolynomialFeatures(degree=2)
        X_poly     = poly.fit_transform(country_df[['year']])
        poly_model = LinearRegression()
        poly_model.fit(X_poly, country_df['primary_energy_consumption'])

        future          = pd.DataFrame({'year': range(2019, 2031)})
        future['forecast'] = poly_model.predict(poly.transform(future))
        past_forecast   = poly_model.predict(poly.transform(country_df[['year']]))

        fig5, ax5 = plt.subplots(figsize=(12, 5))
        fig5.patch.set_facecolor('#0e1117')
        ax5.set_facecolor('#1e2130')
        ax5.plot(country_df['year'], country_df['primary_energy_consumption'],
                 label='Historical Data', color='#74c0fc', marker='o', markersize=4)
        ax5.plot(country_df['year'], past_forecast,
                 label='Model Curve', color='#ffd43b', linewidth=2)
        ax5.plot(future['year'], future['forecast'],
                 label='Forecast (2019-2030)', color='#ff6b6b',
                 marker='o', markersize=5, linestyle='--')
        ax5.axvline(x=country_df['year'].max(), color='#888', linestyle=':', label='Forecast Start')
        ax5.set_xlabel('Year', color='white')
        ax5.set_ylabel('Energy Consumption (TWh)', color='white')
        ax5.tick_params(colors='white')
        ax5.legend(facecolor='#1e2130', labelcolor='white')
        ax5.spines['bottom'].set_color('#444')
        ax5.spines['left'].set_color('#444')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        st.pyplot(fig5)

        st.dataframe(future.rename(columns={'year': 'Year', 'forecast': 'Forecasted Consumption (TWh)'}),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════
# TAB 4 — AI MODELS
# ══════════════════════════════════════════
with tab4:

    # Model comparison
    st.header("🤖 Model Comparison")
    st.write("3 different AI models competing on the same data!")

    col_a, col_b = st.columns(2)
    with col_a:
        selected_region3 = st.selectbox("Select region:", list(regions.keys()), key="b3")
    with col_b:
        region_countries3 = [c for c in regions[selected_region3] if c in df['country'].values]
        selected_country3 = st.selectbox("Select country:", region_countries3, key="u3")

    country_df3 = df[df['country'] == selected_country3].copy()

    if len(country_df3) >= 10:
        X = country_df3[['year', 'population', 'gdp', 'fossil_fuel_consumption', 'renewables_consumption']]
        y = country_df3['primary_energy_consumption']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        models = {
            "📈 Linear Regression": (LinearRegression(), False),
            "🌲 Random Forest":     (RandomForestRegressor(n_estimators=100, random_state=42), False),
            "🧠 Neural Network (MLP)": (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42), True)
        }

        results  = []
        forecasts = {}

        for name, (model, scaled) in models.items():
            model.fit(X_train_s if scaled else X_train, y_train)
            pred = model.predict(X_test_s if scaled else X_test)
            r2  = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            results.append({
                'Model': name,
                'Accuracy (R²)': f"{r2*100:.1f}%",
                'Mean Absolute Error (MAE)': f"{mae:.2f} TWh"
            })
            forecasts[name] = (pred, r2)

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        best_model = max(forecasts.keys(), key=lambda m: forecasts[m][1])
        st.success(f"🏆 Best model: **{best_model}** — Accuracy: **{forecasts[best_model][1]*100:.1f}%**")

        fig6, ax6 = plt.subplots(figsize=(12, 5))
        fig6.patch.set_facecolor('#0e1117')
        ax6.set_facecolor('#1e2130')
        ax6.plot(y_test.values, label='Actual Value', color='white', linewidth=2, marker='o')
        colors6 = ['#00d4ff', '#ffd43b', '#ff6b6b']
        for (name, (pred, _)), color in zip(forecasts.items(), colors6):
            ax6.plot(pred, label=name, linestyle='--', marker='x', color=color)
        ax6.set_xlabel('Test Data Points', color='white')
        ax6.set_ylabel('Energy Consumption (TWh)', color='white')
        ax6.tick_params(colors='white')
        ax6.legend(facecolor='#1e2130', labelcolor='white')
        ax6.spines['bottom'].set_color('#444')
        ax6.spines['left'].set_color('#444')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        st.pyplot(fig6)

        # --- FEATURE IMPORTANCE ---
        st.markdown("---")
        st.header("📊 Which Features Matter Most?")
        st.write("Which data does the AI rely on most when predicting energy consumption? (value between 0 and 1)")

        rf_model = models["🌲 Random Forest"][0]

        features     = ['Year', 'Population', 'GDP', 'Fossil Fuel', 'Renewables']
        importances  = rf_model.feature_importances_

        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=True)

        fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
        fig_imp.patch.set_facecolor('#0e1117')
        ax_imp.set_facecolor('#1e2130')
        ax_imp.barh(importance_df['Feature'], importance_df['Importance'],
                    color='#00d4ff', edgecolor='#0099bb')
        ax_imp.set_xlabel('Importance Score (1.0 = 100%)', color='white')
        ax_imp.set_title('Features the Model Relies On When Making Decisions', color='white')
        ax_imp.tick_params(colors='white')
        ax_imp.spines['bottom'].set_color('#444')
        ax_imp.spines['left'].set_color('#444')
        ax_imp.spines['top'].set_visible(False)
        ax_imp.spines['right'].set_visible(False)
        st.pyplot(fig_imp)

        # --- LIVE SIMULATION (WHAT-IF) ---
        st.markdown("---")
        st.header("🎛️ Live Simulation: Scenario Testing")
        st.write("Adjust the sliders below to create your own future scenario and see the model's real-time prediction.")

        last_record = country_df3.iloc[-1]

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            sim_year = st.slider("Year", min_value=2020, max_value=2050, value=2030, step=1)
            sim_pop_million = st.slider("Population (Million)", min_value=1.0, max_value=1000.0,
                                        value=float(last_record['population'] / 1e6), step=1.0)
            sim_pop = sim_pop_million * 1e6

            sim_gdp_billion = st.slider("GDP (Billion $)", min_value=1.0, max_value=30000.0,
                                         value=float(last_record['gdp'] / 1e9), step=10.0)
            sim_gdp = sim_gdp_billion * 1e9

        with col_s2:
            sim_fossil = st.slider("Fossil Fuel Consumption (TWh)", min_value=0.0,
                                   max_value=float(country_df3['fossil_fuel_consumption'].max() * 2),
                                   value=float(last_record['fossil_fuel_consumption']), step=10.0)
            sim_ren = st.slider("Renewable Energy (TWh)", min_value=0.0,
                                max_value=float(country_df3['renewables_consumption'].max() * 5),
                                value=float(last_record['renewables_consumption']), step=10.0)

        sim_data = pd.DataFrame({
            'year': [sim_year],
            'population': [sim_pop],
            'gdp': [sim_gdp],
            'fossil_fuel_consumption': [sim_fossil],
            'renewables_consumption': [sim_ren]
        })

        # Simülasyon modeli — tüm ülkelerin verisiyle Random Forest
        @st.cache_data
        def train_sim_model():
            df_all = load_data()
            X_all = df_all[['year', 'population', 'gdp',
                            'fossil_fuel_consumption', 'renewables_consumption']]
            y_all = df_all['primary_energy_consumption']
            rf_sim = RandomForestRegressor(n_estimators=200, random_state=42)
            rf_sim.fit(X_all, y_all)
            return rf_sim

        rf_sim_model = train_sim_model()

        sim_prediction = rf_sim_model.predict(sim_data)[0]
        sim_prediction = max(0, sim_prediction)

        

        st.markdown(f"""
        <div style="background-color: #1e2130; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #00d4ff;">
            <h3 style="color: #aaaaaa; margin-bottom: 0;">Estimated Energy Consumption for This Scenario</h3>
            <h1 style="color: #00d4ff; font-size: 3rem; margin-top: 10px;">{sim_prediction:,.0f} TWh</h1>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # ANOMALY DETECTION
    # ══════════════════════════════════════════
    st.markdown("---")
    st.header("🚨 Anomaly Detection")
    st.write("The AI automatically detects unexpected energy consumption changes in the selected country.")

    from sklearn.ensemble import IsolationForest

    anomaly_region = st.selectbox("Select region:", list(regions.keys()), key="anomaly_region")
    anomaly_countries = [c for c in regions[anomaly_region] if c in df['country'].values]
    anomaly_country = st.selectbox("Select country:", anomaly_countries, key="anomaly_country")

    anomaly_df = df[df['country'] == anomaly_country].copy().sort_values('year')

    if len(anomaly_df) >= 10:

        anomaly_df['change'] = anomaly_df['primary_energy_consumption'].pct_change() * 100

        X_anomaly = anomaly_df[['primary_energy_consumption', 'change']].dropna()
        anomaly_df = anomaly_df.loc[X_anomaly.index]

        anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        anomaly_df['anomaly'] = anomaly_model.fit_predict(X_anomaly)
        anomaly_df['anomaly_label'] = anomaly_df['anomaly'].map({1: 'Normal', -1: '🚨 Anomaly'})

        anomaly_count = (anomaly_df['anomaly'] == -1).sum()
        normal_count  = (anomaly_df['anomaly'] == 1).sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Years", len(anomaly_df))
        with col2:
            st.metric("Normal Years", normal_count)
        with col3:
            st.metric("🚨 Anomaly Years", anomaly_count)

        fig_a, ax_a = plt.subplots(figsize=(14, 5))
        fig_a.patch.set_facecolor('#0e1117')
        ax_a.set_facecolor('#1e2130')

        ax_a.plot(anomaly_df['year'], anomaly_df['primary_energy_consumption'],
                  color='#74c0fc', linewidth=2, label='Energy Consumption', zorder=1)

        normal_pts = anomaly_df[anomaly_df['anomaly'] == 1]
        ax_a.scatter(normal_pts['year'], normal_pts['primary_energy_consumption'],
                     color='#51cf66', s=60, label='Normal', zorder=2)

        anomal_pts = anomaly_df[anomaly_df['anomaly'] == -1]
        ax_a.scatter(anomal_pts['year'], anomal_pts['primary_energy_consumption'],
                     color='#ff6b6b', s=150, marker='X', label='🚨 Anomaly', zorder=3)

        for _, row in anomal_pts.iterrows():
            ax_a.annotate(f"{int(row['year'])}",
                          (row['year'], row['primary_energy_consumption']),
                          textcoords="offset points", xytext=(0, 12),
                          color='#ff6b6b', fontsize=9, ha='center', fontweight='bold')

        ax_a.set_title(f'{anomaly_country} — Anomaly Detection', color='white')
        ax_a.set_xlabel('Year', color='white')
        ax_a.set_ylabel('Energy Consumption (TWh)', color='white')
        ax_a.tick_params(colors='white')
        ax_a.legend(facecolor='#1e2130', labelcolor='white')
        ax_a.spines['bottom'].set_color('#444')
        ax_a.spines['left'].set_color('#444')
        ax_a.spines['top'].set_visible(False)
        ax_a.spines['right'].set_visible(False)
        st.pyplot(fig_a)

        if anomaly_count > 0:
            st.subheader("📋 Detected Anomaly Years")
            anomaly_table = anomal_pts[['year', 'primary_energy_consumption', 'change']].copy()
            anomaly_table.columns = ['Year', 'Consumption (TWh)', 'Change (%)']
            anomaly_table['Year'] = anomaly_table['Year'].astype(int)
            anomaly_table['Change (%)'] = anomaly_table['Change (%)'].round(1)
            anomaly_table['Consumption (TWh)'] = anomaly_table['Consumption (TWh)'].round(1)
            st.dataframe(anomaly_table, use_container_width=True, hide_index=True)

            st.info("💡 Anomaly years may reflect wars, economic crises, natural disasters, or major policy changes.")

        st.markdown("---")

        # K-Means Clustering
        st.header("🔵 Clustering Analysis (K-Means)")
        st.write("The AI automatically groups countries by their energy consumption profiles.")

        kmeans_df = df[df['year'] == 2018].copy()
        X_kmeans  = kmeans_df[['primary_energy_consumption', 'fossil_fuel_consumption', 'renewables_consumption']].dropna()
        kmeans_df = kmeans_df.loc[X_kmeans.index]
        scaler2   = StandardScaler()
        X_scaled  = scaler2.fit_transform(X_kmeans)

        n_clusters = st.slider("How many groups?", min_value=2, max_value=6, value=3)
        kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_df['Group'] = kmeans.fit_predict(X_scaled)

        group_consumption = kmeans_df.groupby('Group')['primary_energy_consumption'].mean().sort_values()
        group_names = ['🔵 Low', '🟢 Medium', '🟡 High', '🔴 Very High', '🟣 Extreme', '🟠 Max']
        group_map         = {g: group_names[i] for i, g in enumerate(group_consumption.index)}
        kmeans_df['Group Name'] = kmeans_df['Group'].map(group_map)

        if 'Turkey' in kmeans_df['country'].values:
            tr_group = kmeans_df[kmeans_df['country'] == 'Turkey']['Group Name'].values[0]
            st.info(f"🇹🇷 Turkey's group: **{tr_group}**")

        col1, col2 = st.columns(2)
        with col1:
            for group_name in sorted(kmeans_df['Group Name'].unique()):
                group_countries = kmeans_df[kmeans_df['Group Name'] == group_name]['country'].tolist()
                with st.expander(f"{group_name} — {len(group_countries)} countries"):
                    st.write(", ".join(sorted(group_countries)))

        with col2:
            fig7, ax7 = plt.subplots(figsize=(7, 5))
            fig7.patch.set_facecolor('#0e1117')
            ax7.set_facecolor('#1e2130')
            color_map = {
                '🔵 Low':       '#00d4ff',
                '🟢 Medium':    '#51cf66',
                '🟡 High':      '#ffd43b',
                '🔴 Very High': '#ff6b6b',
                '🟣 Extreme':   '#cc5de8',
                '🟠 Max':       '#ff922b'
            }
            for group_name in sorted(kmeans_df['Group Name'].unique()):
                gv = kmeans_df[kmeans_df['Group Name'] == group_name]
                ax7.scatter(gv['fossil_fuel_consumption'], gv['renewables_consumption'],
                label=group_name, color=color_map[group_name], s=80, alpha=0.8)
            if 'Turkey' in kmeans_df['country'].values:
                tr = kmeans_df[kmeans_df['country'] == 'Turkey']
                ax7.scatter(tr['fossil_fuel_consumption'], tr['renewables_consumption'],
                            color='white', s=250, marker='*', label='🇹🇷 Turkey', zorder=5)
            ax7.set_xlabel('Fossil Fuel (TWh)', color='white')
            ax7.set_ylabel('Renewables (TWh)', color='white')
            ax7.tick_params(colors='white')
            ax7.legend(facecolor='#1e2130', labelcolor='white', fontsize=8)
            ax7.spines['bottom'].set_color('#444')
            ax7.spines['left'].set_color('#444')
            ax7.spines['top'].set_visible(False)
            ax7.spines['right'].set_visible(False)
            st.pyplot(fig7)