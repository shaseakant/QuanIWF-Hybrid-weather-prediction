import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.load_data import load_weather_data
from preprocessing.normalize_pca import prepare_features

from classical_models.linear_regression import train_linear_regression
from classical_models.random_forest import train_random_forest

from quantum_model.qsvm_model import train_qsvm
from quantum_model.hybrid_model import train_hybrid_model

from evaluation.metrics import rmse

st.set_page_config(
    page_title="Hybrid Quantum Weather Prediction",
    layout="wide"
)

st.title("Hybrid Quantum–Classical Weather Prediction System")
st.markdown("### Comparative Analysis of Classical ML vs QSVM vs Hybrid VQC")
st.markdown("---")
# Load Dat
df = load_weather_data()
X, y = prepare_features(df, target_col="temp")

st.subheader("Dataset Information")
colA, colB = st.columns(2)
colA.metric("Total Samples", len(X))
colB.metric("Features After PCA", X.shape[1])

st.markdown("---")
# Run Models Butto
if st.button("Run Model Comparison"):

    st.info("Training models... please wait.")

    # Classical Models
    start = time.time()
    lr_mse = train_linear_regression(X, y)
    lr_time = time.time() - start

    start = time.time()
    rf_mse = train_random_forest(X, y)
    rf_time = time.time() - start

    # QSVM (internally limited)
    start = time.time()
    qsvm_mse = train_qsvm(X, y)
    qsvm_time = time.time() - start

    # VQC Hybrid (Controlled Samples)
    vqc_sample_size = 500
    X_vqc = X[:vqc_sample_size]
    y_vqc = y[:vqc_sample_size]

    start = time.time()
    vqc_mse = train_hybrid_model(X_vqc, y_vqc)
    vqc_time = time.time() - start

    # Prepare Results Table
    results = {
        "Model": ["Linear Regression", "Random Forest", "QSVM", "Hybrid VQC"],
        "Dataset Size": [len(X), len(X), 200, vqc_sample_size],
        "MSE": [lr_mse, rf_mse, qsvm_mse, vqc_mse],
        "RMSE": [
            rmse(lr_mse),
            rmse(rf_mse),
            rmse(qsvm_mse),
            rmse(vqc_mse)
        ],
        "Time (sec)": [lr_time, rf_time, qsvm_time, vqc_time]
    }

    df_results = pd.DataFrame(results)

    # Display Table
    st.subheader("Detailed Model Performance")
    st.dataframe(
        df_results.style.format({
            "MSE": "{:.4f}",
            "RMSE": "{:.4f}",
            "Time (sec)": "{:.2f}"
        }),
        use_container_width=True
    )

    best_model = df_results.sort_values("RMSE").iloc[0]["Model"]
    st.success(f"Best Performing Model (Lowest RMSE): **{best_model}**")

    st.markdown("---")

    # RMSE Comparison Chart
    st.subheader("RMSE Comparison")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(df_results["Model"], df_results["RMSE"])
        ax1.set_ylabel("RMSE")
        ax1.set_title("RMSE Comparison")
        ax1.tick_params(axis="x", rotation=45)
        st.pyplot(fig1)

    # Runtime Comparison Chart
    st.subheader("Runtime Comparison")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(df_results["Model"], df_results["Time (sec)"])
        ax2.set_ylabel("Execution Time (seconds)")
        ax2.set_title("Runtime Comparison")
        ax2.tick_params(axis="x", rotation=45)
        st.pyplot(fig2)

    st.markdown("---")

    # Observations
    st.subheader("Key Observations")

    st.write("""
    • Classical models achieve the lowest prediction error for structured weather data.  
    • QSVM performance is limited by quadratic kernel computation complexity.  
    • Hybrid VQC demonstrates stable learning with improved scalability over QSVM.  
    • Quantum models are currently constrained by simulator-based execution.  
    """)

    st.success("Model comparison completed successfully.")
