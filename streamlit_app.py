# ==============================================================================
# CARBON SHIELD – EU ETS + CBAM Risk Management Tool
# Streamlit Application – Version 3.0 (Enterprise Ready)
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import yfinance as yf
from datetime import datetime
import warnings
import io
import tempfile
import os
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_score, recall_score
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import cm
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
        "default_benchmark": 1.328,
        "imported_materials": {
            "Steel": {"emission_intensity": 1.85, "usage_per_tonne": 0.8},
            "Cement": {"emission_intensity": 0.75, "usage_per_tonne": 0.9},
            "Aluminium": {"emission_intensity": 8.0, "usage_per_tonne": 1.0},
            "Fertilisers": {"emission_intensity": 1.2, "usage_per_tonne": 0.6},
            "Electricity": {"emission_intensity": 0.0, "usage_per_tonne": 0.0},
            "Hydrogen": {"emission_intensity": 0.0, "usage_per_tonne": 0.0}
        }
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

def get_live_eua_price():
    try:
        tickers = CONFIG["eua"]["yfinance_tickers"]
        eur_usd = CONFIG["eua"]["eur_usd"]
        for symbol in tickers:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    price_usd = hist['Close'].iloc[-1]
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

@st.cache_data(ttl=300)
def get_live_eua_price_cached():
    return get_live_eua_price()

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

class ETSComplianceEngine:
    def __init__(self, config):
        self.config = config
        self.base_intensity = config["ets"]["base_energy_intensity"]
        self.lambda_factor = config["ets"]["lambda_intensity_factor"]
        self.shock_factor = config["ets"]["scope1_shock_factor"]

    def calculate(self, production_volume, eua_price, free_allowances,
                  lambda_avg=0.0, shock_multiplier=0.0):
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

class CBAMCalculator:
    def __init__(self, config):
        self.config = config
        self.benchmarks = config["cbam"]["benchmarks"]
        self.imported_data = config["cbam"]["imported_materials"]

    def calculate(self, total_carbon, production_volume, eua_price,
                  benchmark=None, product_category="Steel",
                  imported_intensity=None, imported_usage=None):
        if production_volume <= 0:
            return {
                "embedded_emissions_per_tonne": 0.0,
                "internal_emissions_per_tonne": 0.0,
                "imported_emissions_per_tonne": 0.0,
                "benchmark_used": benchmark or self.benchmarks.get(product_category, 1.0),
                "excess_emissions": 0.0,
                "excess_total": 0.0,
                "cbam_liability": 0.0,
                "product_category": product_category
            }
        internal_emissions = total_carbon / production_volume
        if imported_intensity is None:
            imported_intensity = self.imported_data.get(product_category, {}).get("emission_intensity", 0.0)
        if imported_usage is None:
            imported_usage = self.imported_data.get(product_category, {}).get("usage_per_tonne", 0.0)
        imported_emissions = imported_intensity * imported_usage
        embedded = internal_emissions + imported_emissions
        if benchmark is None:
            benchmark = self.benchmarks.get(product_category, 1.0)
        excess = max(0.0, embedded - benchmark)
        liability = excess * production_volume * eua_price
        return {
            "embedded_emissions_per_tonne": embedded,
            "internal_emissions_per_tonne": internal_emissions,
            "imported_emissions_per_tonne": imported_emissions,
            "benchmark_used": benchmark,
            "excess_emissions": excess,
            "excess_total": excess * production_volume,
            "cbam_liability": liability,
            "product_category": product_category
        }

# ==============================================================================
# 2. ENHANCED SCENARIO FUNCTIONS
# ==============================================================================

def run_single_scenario(engine, cbam, kernel, prod_vol, eua_price, free_alloc,
                        water_base, water_vol, shock_mult, n_steps,
                        cbam_benchmark, product_category,
                        imported_intensity, imported_usage,
                        trigger_threshold, historical_data=None):
    """Run a single simulation scenario with optional historical data."""
    kernel.trigger_threshold = trigger_threshold
    records = []
    
    for t in range(n_steps):
        if historical_data is not None and t < len(historical_data):
            X_t = historical_data[t]
        else:
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
            product_category=product_category,
            imported_intensity=imported_intensity,
            imported_usage=imported_usage
        )

        records.append({
            "time": t,
            "lambda_avg": lambda_avg,
            "lambda_max": np.max(lambda_vec),
            "triggered_any": np.any(triggered),
            **result,
            "internal_emissions_per_tonne": cbam_res["internal_emissions_per_tonne"],
            "imported_emissions_per_tonne": cbam_res["imported_emissions_per_tonne"],
            "embedded_emissions_per_tonne": cbam_res["embedded_emissions_per_tonne"],
            "cbam_liability": cbam_res["cbam_liability"],
            "excess_emissions": cbam_res["excess_emissions"],
            "benchmark_used": cbam_res["benchmark_used"]
        })
    return pd.DataFrame(records)

def run_scenario_analysis(prod_vol, eua_price, free_alloc, n_steps,
                          cbam_benchmark, product_category,
                          imported_intensity, imported_usage,
                          trigger_threshold, historical_data=None):
    """Run 3 scenarios with optional historical data."""
    np.random.seed(42)
    
    engine = ETSComplianceEngine(CONFIG)
    cbam = CBAMCalculator(CONFIG)
    kernel = AramisAlfaPulseKernel(CONFIG)

    scenarios = {
        "Optimistic": {
            "water_base": 2000,
            "water_vol": 200,
            "shock_mult": 0.0
        },
        "Base": {
            "water_base": CONFIG["simulation"]["default_water_base"],
            "water_vol": CONFIG["simulation"]["default_water_vol"],
            "shock_mult": 0.1
        },
        "Pessimistic": {
            "water_base": 5000,
            "water_vol": 800,
            "shock_mult": 0.3
        }
    }

    results = {}
    for name, params in scenarios.items():
        df = run_single_scenario(
            engine, cbam, kernel,
            prod_vol, eua_price, free_alloc,
            params["water_base"],
            params["water_vol"],
            params["shock_mult"],
            n_steps,
            cbam_benchmark, product_category,
            imported_intensity, imported_usage,
            trigger_threshold,
            historical_data
        )
        results[name] = df

    return results

def get_scenario_summary(results):
    """Extract key metrics from scenario results."""
    summary = {}
    for name, df in results.items():
        last = df.iloc[-1]
        summary[name] = {
            "total_carbon": last["total_carbon"],
            "risk_eur": last["risk_eur"],
            "cbam_liability": last["cbam_liability"],
            "lambda_avg": last["lambda_avg"],
            "cascade": last["triggered_any"],
            "embedded": last["embedded_emissions_per_tonne"],
            "internal": last["internal_emissions_per_tonne"],
            "imported": last["imported_emissions_per_tonne"]
        }
    return summary

# ==============================================================================
# 3. HISTORICAL DATA UPLOAD
# ==============================================================================

def upload_historical_data():
    """Upload historical sensor data CSV."""
    st.subheader("📁 Historical Data Upload")
    st.markdown("Upload your own sensor data instead of using generated data.")
    st.info("CSV format: time | component_id | sensor_1 | ... | sensor_K (K=10)")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Format: time, component_id, sensor_1, ..., sensor_K"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} rows")
            
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10))
                st.caption(f"Columns: {', '.join(df.columns)}")
            
            # Convert to numpy array format (n_samples, J, K)
            # Assuming J=4 components, K=10 sensors
            # Data should have columns: time, component_id, sensor_1...sensor_10
            if 'component_id' in df.columns:
                J = df['component_id'].nunique()
                # Group by time and component_id
                sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
                if sensor_cols:
                    K = len(sensor_cols)
                    # Pivot to (time, component, sensor)
                    pivot_df = df.pivot(index='time', columns='component_id', values=sensor_cols)
                    # Convert to numpy array (n_times, J, K)
                    data_array = pivot_df.values.reshape(-1, J, K)
                    st.success(f"✅ Data reshaped to ({data_array.shape[0]}, {J}, {K})")
                    return data_array, df
            else:
                # Try raw numpy format
                data_array = df.values[:, 1:].reshape(-1, 4, 10)
                st.success(f"✅ Data reshaped to ({data_array.shape[0]}, 4, 10)")
                return data_array, df
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return None, None
    return None, None

# ==============================================================================
# 4. MLE CALIBRATION
# ==============================================================================

class MLEKalibrator:
    """Maximum Likelihood Estimation for kernel parameters."""
    
    def __init__(self, config):
        self.config = config
    
    def calibrate(self, sensor_data, fault_labels):
        """
        Calibrate kernel parameters using historical data.
        
        Parameters:
        - sensor_data: (n_samples, J, K) array
        - fault_labels: (n_samples,) boolean array (True = fault occurred)
        
        Returns:
        - optimized parameters
        """
        def negative_log_likelihood(params):
            beta, mu, kick, gamma, sigma = params
            
            kernel = AramisAlfaPulseKernel(self.config)
            kernel.beta = beta
            kernel.mu = mu
            kernel.kick = kick
            kernel.sigma_multiplier = sigma
            kernel.gamma = np.full((kernel.J, kernel.J), gamma)
            np.fill_diagonal(kernel.gamma, 0.0)
            
            predictions = []
            for t in range(len(sensor_data)):
                _, triggered = kernel.process_time_step(sensor_data[t])
                predictions.append(np.any(triggered))
            
            log_likelihood = 0
            for pred, actual in zip(predictions, fault_labels):
                p = pred + 1e-6
                if actual:
                    log_likelihood += np.log(p)
                else:
                    log_likelihood += np.log(1 - p)
            
            return -log_likelihood
        
        initial = [
            self.config["kernel"]["beta"],
            self.config["kernel"]["mu"],
            self.config["kernel"]["kick"],
            self.config["kernel"]["gamma"],
            self.config["kernel"].get("sigma_multiplier", 2.7)
        ]
        
        bounds = [
            (0.01, 0.5),
            (0.01, 0.5),
            (0.01, 0.3),
            (0.01, 0.3),
            (1.5, 4.0)
        ]
        
        with st.spinner("🔬 Running MLE optimization..."):
            result = minimize(
                negative_log_likelihood,
                initial,
                bounds=bounds,
                method='L-BFGS-B'
            )
        
        if result.success:
            # Run final prediction for metrics
            kernel = AramisAlfaPulseKernel(self.config)
            kernel.beta = result.x[0]
            kernel.mu = result.x[1]
            kernel.kick = result.x[2]
            kernel.sigma_multiplier = result.x[4]
            kernel.gamma = np.full((kernel.J, kernel.J), result.x[3])
            np.fill_diagonal(kernel.gamma, 0.0)
            
            predictions = []
            for t in range(len(sensor_data)):
                _, triggered = kernel.process_time_step(sensor_data[t])
                predictions.append(np.any(triggered))
            
            return {
                "beta": result.x[0],
                "mu": result.x[1],
                "kick": result.x[2],
                "gamma": result.x[3],
                "sigma_multiplier": result.x[4],
                "success": True,
                "predictions": predictions
            }
        else:
            return {
                "success": False,
                "message": str(result.message)
            }

# ==============================================================================
# 5. PDF REPORT GENERATOR
# ==============================================================================

def generate_pdf_report(results_df, summary_data, scenario_results=None):
    """Generate professional PDF report."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_path = tmp_file.name
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a5276'),
        alignment=0
    )
    story.append(Paragraph("CARBON SHIELD – EU ETS & CBAM REPORT", title_style))
    story.append(Spacer(1, 0.3*cm))
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#5d6d7e'),
        alignment=0
    )
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("1. EXECUTIVE SUMMARY", styles['Heading2']))
    story.append(Spacer(1, 0.2*cm))
    
    summary_text = f"""
    <b>Key Findings:</b><br/>
    • Total Carbon Footprint: {summary_data.get('total_carbon', 0):,.2f} tCO2e<br/>
    • ETS Financial Risk: €{summary_data.get('risk_eur', 0):,.2f}<br/>
    • CBAM Liability: €{summary_data.get('cbam_liability', 0):,.2f}<br/>
    • Average Lambda: {summary_data.get('lambda_avg', 0):.3f}<br/>
    • Cascade Detected: {'YES' if summary_data.get('cascade', False) else 'NO'}
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    story.append(PageBreak())
    
    # Metrics Table
    story.append(Paragraph("2. KEY METRICS", styles['Heading2']))
    story.append(Spacer(1, 0.2*cm))
    
    metrics_data = [
        ["Metric", "Value"],
        ["Total Carbon Footprint", f"{summary_data.get('total_carbon', 0):,.2f} tCO2e"],
        ["ETS Financial Risk", f"€{summary_data.get('risk_eur', 0):,.2f}"],
        ["CBAM Liability", f"€{summary_data.get('cbam_liability', 0):,.2f}"],
        ["Average Lambda", f"{summary_data.get('lambda_avg', 0):.3f}"],
        ["Cascade Detected", "YES" if summary_data.get('cascade', False) else "NO"],
        ["Internal Emissions", f"{summary_data.get('internal', 0):.3f} tCO2/t"],
        ["Imported Emissions", f"{summary_data.get('imported', 0):.3f} tCO2/t"],
        ["Total Embedded", f"{summary_data.get('embedded', 0):.3f} tCO2/t"]
    ]
    
    table = Table(metrics_data, colWidths=[8*cm, 8*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5*cm))
    story.append(PageBreak())
    
    # PPA Recommendation
    story.append(Paragraph("3. PPA STRATEGY RECOMMENDATION", styles['Heading2']))
    story.append(Spacer(1, 0.2*cm))
    
    ppa_text = f"""
    <b>Recommendation:</b> {summary_data.get('ppa_recommendation', 'WAIT')}<br/>
    <b>Risk without PPA:</b> €{summary_data.get('risk_without_ppa', 0):,.2f}<br/>
    <b>Risk with PPA:</b> €{summary_data.get('risk_with_ppa', 0):,.2f}<br/>
    <b>Total Savings:</b> €{summary_data.get('ppa_savings', 0):,.2f}
    """
    story.append(Paragraph(ppa_text, styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    story.append(PageBreak())
    
    # Scenario Analysis
    if scenario_results:
        story.append(Paragraph("4. SCENARIO ANALYSIS", styles['Heading2']))
        story.append(Spacer(1, 0.2*cm))
        
        scenario_data = [["Scenario", "Risk (EUR)", "Carbon (tCO2e)", "CBAM (EUR)"]]
        for name, metrics in scenario_results.items():
            scenario_data.append([
                name,
                f"{metrics.get('risk_eur', 0):,.2f}",
                f"{metrics.get('total_carbon', 0):,.2f}",
                f"{metrics.get('cbam_liability', 0):,.2f}"
            ])
        
        table = Table(scenario_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.5*cm))
        story.append(PageBreak())
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#5d6d7e'),
        alignment=0
    )
    story.append(Paragraph(
        f"Generated by Carbon Shield v3.0 | Ognjen Raketić, M.Sc. | {datetime.now().strftime('%Y')}",
        footer_style
    ))
    
    doc.build(story)
    return pdf_path

def display_pdf_download_button(pdf_path):
    """Display PDF download button in Streamlit."""
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    st.download_button(
        label="📑 Download PDF Report",
        data=pdf_bytes,
        file_name=f"carbon_shield_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    try:
        os.unlink(pdf_path)
    except:
        pass

# ==============================================================================
# 6. MAIN STREAMLIT APP
# ==============================================================================

def main():
    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state['historical_data'] = None
    if 'calibration_results' not in st.session_state:
        st.session_state['calibration_results'] = None
    if 'current_df' not in st.session_state:
        st.session_state['current_df'] = None
    if 'current_metadata' not in st.session_state:
        st.session_state['current_metadata'] = None
    
    st.title("🛡️ Carbon Shield – EU ETS & CBAM Risk Manager")
    st.markdown("**Professional EU ETS Risk Management System with CBAM Integration**")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Parameters")

        prod_vol = st.number_input(
            "Production Volume (tonnes)",
            value=CONFIG["simulation"]["default_production"],
            min_value=100,
            max_value=50000,
            step=500,
            help="Total production volume in tonnes"
        )

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
            min_value=0,
            max_value=10000,
            step=100,
            help="Free allocation under EU ETS"
        )

        st.markdown("---")

        trigger_threshold = st.slider(
            "Trigger Threshold (Cascade)",
            min_value=0.30,
            max_value=0.85,
            value=CONFIG["kernel"]["trigger_threshold"],
            step=0.05,
            help="Lambda threshold for cascade detection (lower = more sensitive)"
        )

        st.markdown("---")

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

        st.subheader("📊 CBAM Settings")
        product_category = st.selectbox(
            "Product Category",
            CONFIG["cbam"]["product_categories"],
            index=0
        )

        default_benchmark = CONFIG["cbam"]["benchmarks"].get(product_category, 1.0)
        cbam_benchmark = st.number_input(
            "CBAM Benchmark (tCO2/tonne)",
            value=default_benchmark,
            min_value=0.0,
            max_value=3.0,
            step=0.01,
            help="EU benchmark for embedded emissions (lower = stricter)"
        )

        st.subheader("📦 Imported Materials (Precursors)")
        default_intensity = CONFIG["cbam"]["imported_materials"].get(product_category, {}).get("emission_intensity", 0.0)
        default_usage = CONFIG["cbam"]["imported_materials"].get(product_category, {}).get("usage_per_tonne", 0.0)

        imported_intensity = st.number_input(
            "Imported material emission intensity (tCO2e/tonne)",
            value=default_intensity,
            min_value=0.0,
            max_value=10.0,
            step=0.01,
            help="Embedded emissions of the imported raw material (precursor)"
        )
        imported_usage = st.number_input(
            "Imported material usage (tonnes per tonne of product)",
            value=default_usage,
            min_value=0.0,
            max_value=2.0,
            step=0.01,
            help="How many tonnes of imported material are used per tonne of final product"
        )

        n_steps = st.slider(
            "Simulation Steps",
            min_value=10,
            max_value=200,
            value=CONFIG["simulation"]["default_steps"],
            step=10
        )

        st.markdown("---")
        
        # Main buttons
        col1, col2 = st.columns(2)
        with col1:
            run_button = st.button("🚀 Run Simulation", use_container_width=True, type="primary")
        with col2:
            scenario_button = st.button("📊 3 Scenarios", use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Advanced Features")
        
        upload_historical = st.checkbox("📁 Upload Historical Data")
        calibrate_button = st.button("🔬 Calibrate Model", use_container_width=True)
        pdf_report_button = st.button("📑 Generate PDF Report", use_container_width=True)

    # ==============================================================
    # HISTORICAL DATA UPLOAD
    # ==============================================================
    if upload_historical:
        historical_data, raw_df = upload_historical_data()
        if historical_data is not None:
            st.session_state['historical_data'] = historical_data
            st.session_state['historical_raw'] = raw_df

    # ==============================================================
    # MLE CALIBRATION
    # ==============================================================
    if calibrate_button:
        if st.session_state['historical_data'] is not None:
            with st.spinner("🔬 Running MLE calibration..."):
                data = st.session_state['historical_data']
                # Generate synthetic fault labels for demonstration (actual use would have real labels)
                # In practice, user would upload fault labels separately
                fault_labels = np.random.choice([True, False], size=len(data), p=[0.15, 0.85])
                
                calibrator = MLEKalibrator(CONFIG)
                results = calibrator.calibrate(data, fault_labels)
                
                if results['success']:
                    st.session_state['calibration_results'] = results
                    st.success("✅ Calibration successful!")
                    
                    # Update CONFIG with calibrated values
                    CONFIG["kernel"]["beta"] = results["beta"]
                    CONFIG["kernel"]["mu"] = results["mu"]
                    CONFIG["kernel"]["kick"] = results["kick"]
                    CONFIG["kernel"]["gamma"] = results["gamma"]
                    CONFIG["kernel"]["sigma_multiplier"] = results["sigma_multiplier"]
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("β (beta)", f"{results['beta']:.3f}")
                    with col2:
                        st.metric("μ (mu)", f"{results['mu']:.3f}")
                    with col3:
                        st.metric("Kick", f"{results['kick']:.3f}")
                    with col4:
                        st.metric("γ (gamma)", f"{results['gamma']:.3f}")
                    with col5:
                        st.metric("σ (sigma)", f"{results['sigma_multiplier']:.2f}")
                else:
                    st.error(f"❌ Calibration failed: {results.get('message', 'Unknown error')}")
        else:
            st.warning("⚠️ Please upload historical data first.")

    # ==============================================================
    # SINGLE SIMULATION
    # ==============================================================
    if run_button:
        with st.spinner("🧮 Running simulation..."):
            try:
                np.random.seed(42)
                engine = ETSComplianceEngine(CONFIG)
                cbam = CBAMCalculator(CONFIG)
                kernel = AramisAlfaPulseKernel(CONFIG)

                df = run_single_scenario(
                    engine, cbam, kernel,
                    prod_vol, eua_price, free_alloc,
                    CONFIG["simulation"]["default_water_base"],
                    CONFIG["simulation"]["default_water_vol"],
                    0.1, n_steps,
                    cbam_benchmark, product_category,
                    imported_intensity, imported_usage,
                    trigger_threshold,
                    st.session_state['historical_data']
                )

                st.session_state['current_df'] = df
                last = df.iloc[-1]
                
                # Store metadata for PDF
                st.session_state['current_metadata'] = {
                    "total_carbon": last["total_carbon"],
                    "risk_eur": last["risk_eur"],
                    "cbam_liability": last["cbam_liability"],
                    "lambda_avg": last["lambda_avg"],
                    "cascade": last["triggered_any"],
                    "internal": last["internal_emissions_per_tonne"],
                    "imported": last["imported_emissions_per_tonne"],
                    "embedded": last["embedded_emissions_per_tonne"]
                }

                st.markdown("## 📊 Simulation Results")
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

                st.markdown("### 🔍 CBAM Embedded Emissions Breakdown")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Internal emissions", f"{last['internal_emissions_per_tonne']:.3f} tCO2/t")
                with col2:
                    st.metric("Imported (precursor)", f"{last['imported_emissions_per_tonne']:.3f} tCO2/t")
                with col3:
                    st.metric("Total embedded", f"{last['embedded_emissions_per_tonne']:.3f} tCO2/t")

                # PPA Analysis
                st.markdown("---")
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
                consumption = prod_vol * 0.5
                energy_cost_savings = (spot_price - ppa_price) * consumption
                total_savings = savings + energy_cost_savings
                recommendation = "✅ BUY PPA" if total_savings > 0 else "⚠️ WAIT"

                # Store PPA for PDF
                st.session_state['current_metadata']['ppa_recommendation'] = recommendation
                st.session_state['current_metadata']['risk_without_ppa'] = result_no_ppa["risk_eur"]
                st.session_state['current_metadata']['risk_with_ppa'] = result_ppa["risk_eur"]
                st.session_state['current_metadata']['ppa_savings'] = total_savings

                if result_no_ppa["risk_eur"] == 0 and result_ppa["risk_eur"] == 0:
                    st.warning("⚠️ ETS risk is zero. PPA provides no additional ETS benefit.")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Risk without PPA", f"{result_no_ppa['risk_eur']:,.2f} EUR")
                with col2:
                    st.metric("Risk with PPA", f"{result_ppa['risk_eur']:,.2f} EUR")
                with col3:
                    st.metric("Risk Savings", f"{savings:,.2f} EUR")
                with col4:
                    st.metric("Total Savings", f"{total_savings:,.2f} EUR", delta=recommendation)

                if energy_cost_savings != 0:
                    st.caption(f"⚡ Energy cost savings: **{energy_cost_savings:,.2f} EUR**")

                if total_savings > 0:
                    st.success(f"✅ Recommendation: **BUY PPA** – Total savings of **{total_savings:,.2f} EUR**")
                else:
                    st.warning(f"⚠️ Recommendation: **WAIT** – PPA would cost **{abs(total_savings):,.2f} EUR** more than spot")

                # Plots
                st.markdown("---")
                st.subheader("📈 Simulation Charts")
                fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
                fig.subplots_adjust(hspace=0.3)

                axes[0].plot(df["time"], df["lambda_avg"], label="avg lambda", color="blue", linewidth=2)
                axes[0].plot(df["time"], df["lambda_max"], label="max lambda", color="red", linestyle="--", linewidth=1.5)
                axes[0].axhline(y=trigger_threshold, color="gray", linestyle=":", label="trigger threshold", linewidth=1)
                axes[0].set_ylabel("Lambda", fontsize=11)
                axes[0].legend(loc="upper right")
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(df["time"], df["total_carbon"], label="Total Carbon (tCO2e)", color="green", linewidth=2)
                axes[1].set_ylabel("tCO2e", fontsize=11)
                axes[1].legend(loc="upper right")
                axes[1].grid(True, alpha=0.3)

                axes[2].plot(df["time"], df["risk_eur"], label="Financial Risk (EUR)", color="magenta", linewidth=2)
                axes[2].set_ylabel("EUR", fontsize=11)
                axes[2].legend(loc="upper right")
                axes[2].grid(True, alpha=0.3)

                axes[3].plot(df["time"], df["cbam_liability"], label="CBAM Liability (EUR)", color="orange", linewidth=2)
                axes[3].set_ylabel("EUR", fontsize=11)
                axes[3].legend(loc="upper right")
                axes[3].grid(True, alpha=0.3)

                axes[4].plot(df["time"], df["internal_emissions_per_tonne"], label="Internal", color="blue", linewidth=2)
                axes[4].plot(df["time"], df["imported_emissions_per_tonne"], label="Imported (precursor)", color="red", linestyle="--", linewidth=2)
                axes[4].plot(df["time"], df["embedded_emissions_per_tonne"], label="Total embedded", color="green", linewidth=2)
                axes[4].axhline(y=last["benchmark_used"], color="gray", linestyle=":", label="CBAM benchmark", linewidth=1)
                axes[4].set_xlabel("Time step", fontsize=11)
                axes[4].set_ylabel("tCO2 / tonne", fontsize=11)
                axes[4].legend(loc="upper right")
                axes[4].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Knowledge Graph
                st.markdown("---")
                st.subheader("🧠 Knowledge Graph – Semantic Linkage")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                G = nx.DiGraph()
                edges = [
                    ("Water Meter", "Energy Penalty"),
                    ("Energy Penalty", "Carbon Footprint"),
                    ("EUA Price", "ETS Risk"),
                    ("Carbon Footprint", "ETS Risk"),
                    ("Carbon Footprint", "Internal Emissions"),
                    ("Imported Material", "Imported Emissions"),
                    ("Internal Emissions", "CBAM Liability"),
                    ("Imported Emissions", "CBAM Liability"),
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

                # Download
                st.markdown("---")
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

    # ==============================================================
    # SCENARIO ANALYSIS (3 SCENARIOS)
    # ==============================================================
    if scenario_button:
        with st.spinner("🧮 Running 3 scenario analysis..."):
            try:
                results = run_scenario_analysis(
                    prod_vol, eua_price, free_alloc, n_steps,
                    cbam_benchmark, product_category,
                    imported_intensity, imported_usage,
                    trigger_threshold,
                    st.session_state['historical_data']
                )

                summary = get_scenario_summary(results)

                st.markdown("## 📊 3-Scenario Analysis")
                st.markdown("*Optimistic, Base, and Pessimistic scenarios compared*")
                st.markdown("---")

                # Scenario Comparison Table
                st.markdown("### 📋 Scenario Comparison Table")

                scenario_data = []
                for name, metrics in summary.items():
                    scenario_data.append({
                        "Scenario": name,
                        "Carbon (tCO2e)": f"{metrics['total_carbon']:,.2f}",
                        "ETS Risk (EUR)": f"{metrics['risk_eur']:,.2f}",
                        "CBAM Liability (EUR)": f"{metrics['cbam_liability']:,.2f}",
                        "Avg Lambda": f"{metrics['lambda_avg']:.3f}",
                        "Cascade": "🔴 YES" if metrics['cascade'] else "🟢 NO",
                        "Embedded (tCO2/t)": f"{metrics['embedded']:.3f}"
                    })

                df_scenario = pd.DataFrame(scenario_data)
                st.dataframe(df_scenario, use_container_width=True, hide_index=True)

                st.markdown("---")

                # Risk Range
                st.markdown("### 📈 Risk Range & Confidence")

                risk_values = [summary[s]['risk_eur'] for s in summary]
                carbon_values = [summary[s]['total_carbon'] for s in summary]
                cbam_values = [summary[s]['cbam_liability'] for s in summary]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "ETS Risk Range",
                        f"€{min(risk_values):,.0f} – €{max(risk_values):,.0f}",
                        delta=f"±{((max(risk_values)-min(risk_values))/2):,.0f} EUR",
                        delta_color="off"
                    )
                with col2:
                    st.metric(
                        "Carbon Range",
                        f"{min(carbon_values):,.0f} – {max(carbon_values):,.0f} tCO2e",
                        delta=f"±{((max(carbon_values)-min(carbon_values))/2):,.0f} tCO2e",
                        delta_color="off"
                    )
                with col3:
                    st.metric(
                        "CBAM Range",
                        f"€{min(cbam_values):,.0f} – €{max(cbam_values):,.0f}",
                        delta=f"±{((max(cbam_values)-min(cbam_values))/2):,.0f} EUR",
                        delta_color="off"
                    )

                st.markdown("---")

                # Bar Chart
                st.markdown("### 📊 Visual Comparison")
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                scenarios_names = list(summary.keys())
                risk_vals = [summary[s]['risk_eur'] for s in scenarios_names]
                colors_bar = ['#2ecc71', '#3498db', '#e74c3c']

                axes[0].bar(scenarios_names, risk_vals, color=colors_bar)
                axes[0].set_title('ETS Financial Risk', fontsize=12)
                axes[0].set_ylabel('EUR')
                axes[0].grid(True, alpha=0.3)

                carbon_vals = [summary[s]['total_carbon'] for s in scenarios_names]
                axes[1].bar(scenarios_names, carbon_vals, color=colors_bar)
                axes[1].set_title('Total Carbon Footprint', fontsize=12)
                axes[1].set_ylabel('tCO2e')
                axes[1].grid(True, alpha=0.3)

                cbam_vals = [summary[s]['cbam_liability'] for s in scenarios_names]
                axes[2].bar(scenarios_names, cbam_vals, color=colors_bar)
                axes[2].set_title('CBAM Liability', fontsize=12)
                axes[2].set_ylabel('EUR')
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("---")

                # Executive Recommendation
                st.markdown("### 💡 Executive Recommendation")

                base_risk = summary["Base"]["risk_eur"]
                pess_risk = summary["Pessimistic"]["risk_eur"]

                if base_risk > 0:
                    savings_vs_pess = pess_risk - base_risk
                    if savings_vs_pess > 10000:
                        recommendation = f"✅ **Strong opportunity**: Moving from Pessimistic to Base scenario saves **€{savings_vs_pess:,.0f}**. Focus on operational stability."
                    elif savings_vs_pess > 5000:
                        recommendation = f"🟡 **Moderate opportunity**: Moving from Pessimistic to Base scenario saves **€{savings_vs_pess:,.0f}**. Consider targeted interventions."
                    else:
                        recommendation = f"ℹ️ **Limited opportunity**: Base scenario risk is already close to Optimistic. Focus on maintaining current performance."
                else:
                    recommendation = "✅ **Excellent position**: Base scenario ETS risk is zero. Focus on CBAM optimization and PPA strategy."

                st.info(recommendation)

                st.markdown("---")

                # Export all scenarios
                st.markdown("### 📥 Export All Scenarios")
                all_scenarios_df = pd.DataFrame({
                    "Scenario": scenarios_names,
                    "Risk (EUR)": risk_vals,
                    "Carbon (tCO2e)": carbon_vals,
                    "CBAM (EUR)": cbam_vals,
                    "Lambda": [summary[s]['lambda_avg'] for s in scenarios_names],
                    "Cascade": ["YES" if summary[s]['cascade'] else "NO" for s in scenarios_names]
                })

                csv_all = all_scenarios_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download 3-Scenario Results (CSV)",
                    data=csv_all,
                    file_name=f"scenario_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.success("✅ 3-Scenario analysis completed successfully!")

            except Exception as e:
                st.error(f"❌ Error during scenario analysis: {str(e)}")
                st.exception(e)

    # ==============================================================
    # PDF REPORT GENERATOR
    # ==============================================================
    if pdf_report_button:
        if st.session_state['current_df'] is not None and st.session_state['current_metadata'] is not None:
            with st.spinner("📑 Generating PDF report..."):
                try:
                    # Get scenario results if available
                    scenario_results = None
                    if 'scenario_results' in st.session_state:
                        scenario_results = st.session_state['scenario_results']
                    
                    pdf_path = generate_pdf_report(
                        st.session_state['current_df'],
                        st.session_state['current_metadata'],
                        scenario_results
                    )
                    display_pdf_download_button(pdf_path)
                    st.success("✅ PDF report generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
        else:
            st.warning("⚠️ Please run a simulation first before generating a PDF report.")

if __name__ == "__main__":
    main()
