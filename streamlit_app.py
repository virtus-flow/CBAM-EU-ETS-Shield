# ==============================================================================
# CARBON SHIELD – EU ETS + CBAM Risk Management Tool
# Streamlit Application – Version 2.6 (Production Ready)
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Carbon Shield – EU ETS & CBAM Risk Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
CONFIG = {
    "kernel": {
        "n_components": 4,
        "n_sensors": 10,
        "beta": 0.12,
        "mu": 0.18,
        "kick": 0.20,
        "gamma": 0.15,
        "lambda_min": 0.10,
        "lambda_max": 2.0,
        "trigger_threshold": 0.65,
        "reset_after": 30,
        "sigma_multiplier": 2.6
    },
    "ets": {
        "base_energy_intensity": 0.40,
        "lambda_intensity_factor": 1.0,
        "scope1_shock_factor": 0.10
    },
    "eua": {
        "yfinance_tickers": ["KEUA", "KRBN", "CTWO"],
        "fallback_price": 70.0,
        "eur_usd": 0.92,
        "alert_threshold": 85.0
    },
    "cbam": {
        "product_categories": ["Steel", "Cement", "Aluminium", "Fertilisers", "Electricity", "Hydrogen"],
        "default_category": "Steel",
        "benchmarks": {
            "Steel": 1.328,
            "Cement": 0.766,
            "Aluminium": 1.0,
            "Fertilisers": 1.0,
            "Electricity": 0.0,
            "Hydrogen": 0.0
        },
        "default_benchmark": 1.328
    },
    "simulation": {
        "default_steps": 50,
        "default_production": 15000,
        "default_free_allowances": 2200,
        "default_water_base": 3000,
        "default_water_vol": 500
    }
}

# ==============================================================================
# 1. CORE CLASSES
# ==============================================================================

# ---- EUA Price Fetcher (sa keširanjem) ----
def get_live_eua_price():
    """Fetch live EUA price from Yahoo Finance."""
    try:
        tickers = CONFIG["eua"]["yfinance_tickers"]
        eur_usd = CONFIG["eua"]["eur_usd"]
        for symbol in tickers:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    price_usd = hist['Close'].iloc[-1]
                    # Try live EUR/USD
                    try:
                        eur_ticker = yf.Ticker("EURUSD=X")
                        eur_hist = eur_ticker.history(period="1d")
                        if not eur_hist.empty:
                            eur_usd = eur_hist['Close'].iloc[-1]
                    except:
                        pass
                    return price_usd * eur_usd
            except:
                continue
    except:
        pass
    return CONFIG["eua"]["fallback_price"]

@st.cache_data(ttl=300)  # Keširaj 5 minuta
def get_live_eua_price_cached():
    """Cached version of EUA price fetcher."""
    return get_live_eua_price()

# ---- Stochastic Kernel ----
class AramisAlfaPulseKernel:
    def __init__(self, config):
        self.config = config
        self.J = config["kernel"]["n_components"]
        self.K = config["kernel"]["n_sensors"]
        self.beta = config["kernel"]["beta"]
        self.mu = config["kernel"]["mu"]
        self.kick = config["kernel"]["kick"]
        self.lambda_min = config["kernel"]["lambda_min"]
        self.lambda_max = config["kernel"]["lambda_max"]
        self.trigger_threshold = config["kernel"]["trigger_threshold"]
        self.reset_after = config["kernel"]["reset_after"]
        self.sigma_multiplier = config["kernel"].get("sigma_multiplier", 2.7)
        self.gamma = np.full((self.J, self.J), config["kernel"]["gamma"])
        np.fill_diagonal(self.gamma, 0.0)
        self.lambda_vector = np.full(self.J, self.mu)
        self.running_mean = np.zeros((self.J, self.K))
        self.running_M2 = np.zeros((self.J, self.K))
        self.sample_count = np.zeros(self.J, dtype=int)

    def process_time_step(self, X_t):
        tau_hat_triggered = np.zeros(self.J, dtype=bool)
        for j in range(self.J):
            self.sample_count[j] += 1
            if self.sample_count[j] > self.reset_after:
                self.sample_count[j] = 1
                self.running_M2[j, :] = 0.05
            delta = X_t[j, :] - self.running_mean[j, :]
            self.running_mean[j, :] += delta / self.sample_count[j]
            self.running_M2[j, :] += delta * (X_t[j, :] - self.running_mean[j, :])
            stdev = np.sqrt(self.running_M2[j, :] / self.sample_count[j]) + 1e-6
            threshold = self.running_mean[j, :] + self.sigma_multiplier * stdev
            self.lambda_vector[j] += self.beta * (self.mu - self.lambda_vector[j])
            if np.any(X_t[j, :] > threshold):
                shock = self.kick * np.max(X_t[j, :] / (threshold + 1e-6))
                self.lambda_vector[j] += shock
                for i in range(self.J):
                    if i != j:
                        self.lambda_vector[i] += self.gamma[i, j] * self.kick
            self.lambda_vector[j] = np.clip(self.lambda_vector[j], self.lambda_min, self.lambda_max)
            if self.lambda_vector[j] > self.trigger_threshold:
                tau_hat_triggered[j] = True
        return self.lambda_vector.copy(), tau_hat_triggered

# ---- ETS Engine ----
class ETSComplianceEngine:
    def __init__(self, config):
        self.config = config
        self.base_intensity = config["ets"]["base_energy_intensity"]
        self.lambda_factor = config["ets"]["lambda_intensity_factor"]
        self.shock_factor = config["ets"]["scope1_shock_factor"]
    
    def calculate(self, production_volume, eua_price, free_allowances,
                  lambda_avg=0.0, shock_multiplier=0.0):
        # Input validation
        if production_volume <= 0:
            raise ValueError("Production volume must be > 0")
        if eua_price <= 0:
            raise ValueError("EUA price must be > 0")
            
        intensity = self.base_intensity * (1 + lambda_avg * self.lambda_factor)
        scope1_shock = shock_multiplier * production_volume / 10000.0
        A = np.array([[0.05, 0.10, 0.20],
                      [0.15, 0.02, intensity],
                      [0.00, 0.00, 0.01]])
        f = np.array([0.0, 0.0, production_volume])
        x = np.linalg.solve(np.eye(3) - A, f)
        B_scope1 = np.array([0.05, 0.00, 0.20])
        scope1 = np.dot(B_scope1, x) * (1 + scope1_shock)
        scope2 = np.dot(np.array([0.00, 0.38, 0.00]), x)
        total_carbon = scope1 + scope2
        risk_eur = max(0.0, scope1 - free_allowances) * eua_price
        return {
            "scope1": scope1,
            "scope2": scope2,
            "total_carbon": total_carbon,
            "risk_eur": risk_eur
        }

# ---- CBAM Calculator ----
class CBAMCalculator:
    def __init__(self, config):
        self.config = config
        self.benchmarks = config["cbam"]["benchmarks"]
    
    def calculate(self, total_carbon, production_volume, eua_price,
                  benchmark=None, product_category="Steel"):
        if production_volume <= 0:
            return {
                "embedded_emissions_per_tonne": 0.0,
                "benchmark_used": benchmark or self.benchmarks.get(product_category, 1.0),
                "excess_emissions": 0.0,
                "excess_total": 0.0,
                "cbam_liability": 0.0,
                "product_category": product_category
            }
        if benchmark is None:
            benchmark = self.benchmarks.get(product_category, 1.0)
        embedded = total_carbon / production_volume
        excess = max(0.0, embedded - benchmark)
        liability = excess * production_volume * eua_price
        return {
            "embedded_emissions_per_tonne": embedded,
            "benchmark_used": benchmark,
            "excess_emissions": excess,
            "excess_total": excess * production_volume,
            "cbam_liability": liability,
            "product_category": product_category
        }

# ==============================================================================
# 2. STREAMLIT APP
# ==============================================================================

def main():
    # Header
    st.title("🛡️ Carbon Shield – EU ETS & CBAM Risk Manager")
    st.markdown("**Professional EU ETS Risk Management System with CBAM Integration**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        # Production & ETS
        prod_vol = st.number_input(
            "Production Volume (tonnes)",
            value=CONFIG["simulation"]["default_production"],
            min_value=5000,
            max_value=50000,
            step=500,
            help="Total production volume in tonnes"
        )
        
        # Live EUA price with caching
        live_price = get_live_eua_price_cached()
        default_eua = live_price if live_price is not None else CONFIG["eua"]["fallback_price"]
        eua_price = st.number_input(
            "EUA Price (EUR/tCO2e)",
            value=default_eua,
            min_value=30.0,
            max_value=150.0,
            step=1.0,
            help="Current EU Allowance price (auto-updates from Yahoo Finance)"
        )
        
        free_alloc = st.number_input(
            "Free Allowances (tCO2e)",
            value=CONFIG["simulation"]["default_free_allowances"],
            min_value=1000,
            max_value=10000,
            step=100,
            help="Free allocation under EU ETS"
        )
        
        st.markdown("---")
        
        # Sensitivity
        trigger_threshold = st.slider(
            "Trigger Threshold (Cascade)",
            min_value=0.30,
            max_value=0.85,
            value=CONFIG["kernel"]["trigger_threshold"],
            step=0.05,
            help="Lambda threshold for cascade detection (lower = more sensitive)"
        )
        
        # Water / Anomaly
        st.subheader("🌊 Water Emissions")
        water_base = st.slider(
            "Water Emission Base (m³)",
            min_value=500,
            max_value=15000,
            value=CONFIG["simulation"]["default_water_base"],
            step=500
        )
        water_vol = st.slider(
            "Water Volatility",
            min_value=100,
            max_value=1500,
            value=CONFIG["simulation"]["default_water_vol"],
            step=100
        )
        shock_mult = st.slider(
            "Shock Multiplier",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.05
        )
        
        st.markdown("---")
        
        # PPA
        st.subheader("⚡ PPA Strategy")
        spot_price = st.slider(
            "Spot Energy Price (EUR/MWh)",
            min_value=60.0,
            max_value=120.0,
            value=85.0,
            step=1.0
        )
        ppa_price = st.slider(
            "PPA Price (EUR/MWh)",
            min_value=60.0,
            max_value=120.0,
            value=90.0,
            step=1.0
        )
        
        st.markdown("---")
        
        # CBAM
        st.subheader("📊 CBAM Settings")
        product_category = st.selectbox(
            "Product Category",
            CONFIG["cbam"]["product_categories"],
            index=0
        )
        # Dynamic benchmark update
        default_benchmark = CONFIG["cbam"]["benchmarks"].get(product_category, 1.0)
        cbam_benchmark = st.number_input(
            "CBAM Benchmark (tCO2/tonne)",
            value=default_benchmark,
            min_value=0.0,
            max_value=3.0,
            step=0.01,
            help="EU benchmark for embedded emissions (lower = stricter)"
        )
        
        # Simulation steps
        n_steps = st.slider(
            "Simulation Steps",
            min_value=10,
            max_value=200,
            value=CONFIG["simulation"]["default_steps"],
            step=10
        )
        
        st.markdown("---")
        run_button = st.button("🚀 Run Simulation", use_container_width=True, type="primary")

    # Main area
    if run_button:
        with st.spinner("🧮 Running simulation..."):
            try:
                # Set random seed for reproducibility
                np.random.seed(42)
                
                # Instantiate components
                kernel = AramisAlfaPulseKernel(CONFIG)
                engine = ETSComplianceEngine(CONFIG)
                cbam = CBAMCalculator(CONFIG)
                
                # Apply dynamic threshold
                kernel.trigger_threshold = trigger_threshold
                
                # Run simulation
                records = []
                for t in range(n_steps):
                    base_signal = water_base / 1000.0
                    noise = np.random.normal(0, water_vol, size=(kernel.J, kernel.K))
                    X_t = np.maximum(base_signal + noise, 0.0)
                    
                    lambda_vec, triggered = kernel.process_time_step(X_t)
                    lambda_avg = np.mean(lambda_vec)
                    shock = shock_mult + (0.10 if np.any(triggered) else 0.0)
                    
                    result = engine.calculate(
                        prod_vol, eua_price, free_alloc,
                        lambda_avg=lambda_avg,
                        shock_multiplier=shock
                    )
                    
                    cbam_res = cbam.calculate(
                        result["total_carbon"],
                        prod_vol,
                        eua_price,
                        benchmark=cbam_benchmark,
                        product_category=product_category
                    )
                    
                    records.append({
                        "time": t,
                        "lambda_avg": lambda_avg,
                        "lambda_max": np.max(lambda_vec),
                        "triggered_any": np.any(triggered),
                        **result,
                        "embedded_emissions_per_tonne": cbam_res["embedded_emissions_per_tonne"],
                        "cbam_liability": cbam_res["cbam_liability"],
                        "excess_emissions": cbam_res["excess_emissions"],
                        "benchmark_used": cbam_res["benchmark_used"]
                    })
                
                df = pd.DataFrame(records)
                last = df.iloc[-1]
                
                # ----- RESULTS DISPLAY -----
                st.markdown("## 📊 Simulation Results")
                
                # Key metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Carbon", f"{last['total_carbon']:,.2f} tCO2e")
                with col2:
                    st.metric("ETS Financial Risk", f"{last['risk_eur']:,.2f} EUR")
                with col3:
                    st.metric("CBAM Liability", f"{last['cbam_liability']:,.2f} EUR")
                with col4:
                    st.metric("Avg Lambda", f"{last['lambda_avg']:.3f}")
                with col5:
                    cascade_status = "🔴 YES" if last['triggered_any'] else "🟢 NO"
                    st.metric("Cascade Detected", cascade_status)
                
                st.markdown("---")
                
                # PPA Analysis
                st.subheader("💡 PPA Strategy Recommendation")
                
                result_no_ppa = engine.calculate(
                    prod_vol, eua_price, free_alloc,
                    lambda_avg=0.5, shock_multiplier=0.0
                )
                result_ppa = engine.calculate(
                    prod_vol, eua_price, free_alloc,
                    lambda_avg=0.2, shock_multiplier=-0.1
                )
                
                savings = result_no_ppa["risk_eur"] - result_ppa["risk_eur"]
                consumption = prod_vol * 0.5  # MWh
                energy_cost_savings = (spot_price - ppa_price) * consumption
                total_savings = savings + energy_cost_savings
                recommendation = "✅ BUY PPA" if total_savings > 0 else "⚠️ WAIT"
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Risk without PPA", f"{result_no_ppa['risk_eur']:,.2f} EUR")
                with col2:
                    st.metric("Risk with PPA", f"{result_ppa['risk_eur']:,.2f} EUR")
                with col3:
                    st.metric("Risk Savings", f"{savings:,.2f} EUR")
                with col4:
                    st.metric("Total Savings", f"{total_savings:,.2f} EUR", delta=recommendation)
                
                if total_savings > 0:
                    st.success(f"✅ Recommendation: **BUY PPA** – Total savings of **{total_savings:,.2f} EUR**")
                    st.info(f"💡 Break-even PPA price: **{spot_price - (savings / consumption):.2f} EUR/MWh**")
                else:
                    st.warning(f"⚠️ Recommendation: **WAIT** – PPA would cost **{abs(total_savings):,.2f} EUR** more than spot")
                
                st.markdown("---")
                
                # Plots
                st.subheader("📈 Simulation Charts")
                
                # Create four plots
                fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
                fig.subplots_adjust(hspace=0.3)
                
                # Lambda
                axes[0].plot(df["time"], df["lambda_avg"], label="avg lambda", color="blue", linewidth=2)
                axes[0].plot(df["time"], df["lambda_max"], label="max lambda", color="red", linestyle="--", linewidth=1.5)
                axes[0].axhline(y=trigger_threshold, color="gray", linestyle=":", label="trigger threshold", linewidth=1)
                axes[0].set_ylabel("Lambda", fontsize=11)
                axes[0].legend(loc="upper right")
                axes[0].grid(True, alpha=0.3)
                axes[0].set_ylim(0, max(1.0, df["lambda_max"].max() * 1.1))
                
                # Carbon
                axes[1].plot(df["time"], df["total_carbon"], label="Total Carbon (tCO2e)", color="green", linewidth=2)
                axes[1].set_ylabel("tCO2e", fontsize=11)
                axes[1].legend(loc="upper right")
                axes[1].grid(True, alpha=0.3)
                
                # Risk
                axes[2].plot(df["time"], df["risk_eur"], label="Financial Risk (EUR)", color="magenta", linewidth=2)
                axes[2].set_ylabel("EUR", fontsize=11)
                axes[2].legend(loc="upper right")
                axes[2].grid(True, alpha=0.3)
                
                # CBAM
                axes[3].plot(df["time"], df["cbam_liability"], label="CBAM Liability (EUR)", color="orange", linewidth=2)
                axes[3].set_xlabel("Time step", fontsize=11)
                axes[3].set_ylabel("EUR", fontsize=11)
                axes[3].legend(loc="upper right")
                axes[3].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                st.markdown("---")
                
                # Knowledge Graph
                st.subheader("🧠 Knowledge Graph – Semantic Linkage")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                G = nx.DiGraph()
                edges = [
                    ("Water Meter", "Energy Penalty"),
                    ("Energy Penalty", "Carbon Footprint"),
                    ("EUA Price", "ETS Risk"),
                    ("Carbon Footprint", "ETS Risk"),
                    ("Carbon Footprint", "CBAM Liability"),
                    ("CBAM Benchmark", "CBAM Liability"),
                    ("PPA Strategy", "Energy Cost"),
                    ("Energy Cost", "ETS Risk")
                ]
                G.add_edges_from(edges)
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_color="lightblue",
                        node_size=3000, font_size=10, arrowsize=20, 
                        edge_color="gray", ax=ax2, font_weight="bold")
                ax2.set_title("Knowledge Graph: Carbon Shield", fontsize=14)
                st.pyplot(fig2)
                plt.close(fig2)
                
                st.markdown("---")
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Simulation Data (CSV)",
                    data=csv,
                    file_name=f"carbon_shield_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.success("✅ Simulation completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during simulation: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
