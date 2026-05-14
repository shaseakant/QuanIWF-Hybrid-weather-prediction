from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def prepare_features(df, target_col="temp", n_components=4):

    #Time Dependency)
    df = df.copy()
    df["temp_lag1"] = df[target_col].shift(1)

    # Drop first row created due to lag
    df = df.dropna()

    # 2. Selected Features including lag
    features = [
        "tempmax",
        "tempmin",
        "humidity",
        "windspeed",
        "sealevelpressure",
        "temp_lag1"
    ]

    X = df[features].values
    y = df[target_col].values

    # 3. Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. PCA 
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y
