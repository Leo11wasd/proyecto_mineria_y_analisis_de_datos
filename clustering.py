import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
KMEANS_RS=42
def fit_kmeans(df, columna="Club", CANT_CLUSTERS=5):
    target = "transfer_value_estimado"
    
    num_cols = df.select_dtypes(include=np.number).columns
    num_cols = [col for col in num_cols if col != target]

    grouped = df.groupby(columna)[num_cols]
    df_new = grouped.agg(['mean', 'std'])
    df_new.columns = ['{}_{}'.format(col, stat) for col, stat in df_new.columns]
    df_new = df_new.reset_index()

    X = df_new.drop(columns=[columna]).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=CANT_CLUSTERS, random_state=KMEANS_RS)
    clusters = kmeans.fit_predict(X_scaled)

    df_new["cluster"] = clusters

    # guardamos mapping
    mapping = dict(zip(df_new[columna], df_new["cluster"]))

    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "mapping": mapping,
        "columna": columna,
        "cant_clusters": columna,
        
    }
    
    
    
def apply_kmeans(df, model):
    columna = model["columna"]
    mapping = model["mapping"]

    df = df.copy()
    df[f"{columna}_cluster"] = df[columna].map(mapping)

    # opcional: manejar desconocidos
    df[f"{columna}_cluster"] = df[f"{columna}_cluster"].fillna(-1)

    return df