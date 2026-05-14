from preprocessing.load_data import load_weather_data
from preprocessing.normalize_pca import prepare_features

from classical_models.linear_regression import train_linear_regression
from classical_models.random_forest import train_random_forest

from quantum_model.qsvm_model import train_qsvm
from quantum_model.hybrid_model import train_hybrid_model

from evaluation.metrics import rmse
from evaluation.plots import plot_comparison

import time

# 1. Load Weather Dataset
df = load_weather_data()


# 2. Preprocessing + PCA
X, y = prepare_features(df, target_col="temp")

# 3. Classical Models 
print("Training Classical Models...")
lr_mse = train_linear_regression(X, y)
rf_mse = train_random_forest(X, y)


# 4. QSVM 
print("Training QSVM...")
start = time.time()
qsvm_mse = train_qsvm(X, y)
print("QSVM Time:", round(time.time() - start, 4), "seconds")


# 5. VQC 
print("Training VQC Hybrid...")

vqc_sample_size = 500  # <-- change data samples
X_vqc = X[:vqc_sample_size]
y_vqc = y[:vqc_sample_size]

start = time.time()
#vqc_mse = train_hybrid_model(X, y)  # <-- full data samples
vqc_mse = train_hybrid_model(X_vqc, y_vqc)
print("VQC Time:", round(time.time() - start, 4), "seconds")

# 6. Results
print("\n----- FINAL RESULTS -----")
print("Linear Regression  MSE:", lr_mse, "RMSE:", rmse(lr_mse))
print("Random Forest     MSE:", rf_mse, "RMSE:", rmse(rf_mse))
print("QSVM              MSE:", qsvm_mse, "RMSE:", rmse(qsvm_mse))
print("VQC Hybrid        MSE:", vqc_mse, "RMSE:", rmse(vqc_mse))

# 7. Visualization
plot_comparison(
    ["Linear Regression", "Random Forest", "QSVM", "VQC"],
    [lr_mse, rf_mse, qsvm_mse, vqc_mse]
)
