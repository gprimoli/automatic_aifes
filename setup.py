import pandas as pd
import numpy as np
import warnings
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.utils import resample

## Configuration parameters
n_nodes = 5
base_ip = "172.19.0."
ip_nodes = [f"{base_ip}{i+2}" for i in range(n_nodes)]
for i in range(n_nodes):
    try:
        os.mkdir(f'dataset/{ip_nodes[i]}')
    except:
        print("Folder already created.")

def split_and_save_csv(X, y, n_parts, prefix):
    base_chunk_size = len(X) // n_parts
    remainder = len(X) % n_parts
    start_idx = 0
    for i in range(n_parts):
        chunk_size = base_chunk_size + 1 if i < remainder else base_chunk_size
        end_idx = start_idx + chunk_size
        X_chunk = X.iloc[start_idx:end_idx]
        y_chunk = y.iloc[start_idx:end_idx]
        X_chunk.to_csv(f'dataset/{ip_nodes[i]}/x_{prefix}.csv', index=False, header=False)
        y_chunk.to_csv(f'dataset/{ip_nodes[i]}/y_{prefix}.csv', index=False, header=False)
        print(f"Part {i+1} saved as {prefix}_{ip_nodes[i]}_training.csv")
        
        start_idx = end_idx

def split_and_save_csv_non_iid(X, y, n_parts, prefix):
    y_labels = y.idxmax(axis=1)  # trova l'indice della classe per ciascun esempio one-hot
    data = X.copy()
    data['Label'] = y_labels

    unique_labels = y_labels.unique()
    n_labels = len(unique_labels)
    
    # Distribuzione delle classi in modo che ogni nodo abbia una o due classi predominanti
    class_splits = {i: [] for i in range(n_parts)}
    for i, label in enumerate(unique_labels):
        indices = data[data['Label'] == label].index.tolist()
        np.random.shuffle(indices)
        chunk_size = len(indices) // n_parts
        for j in range(n_parts):
            selected = indices[j*chunk_size:(j+1)*chunk_size]
            class_splits[j].extend(selected)

    # Per ogni nodo salva i dati corrispondenti
    for i in range(n_parts):
        idxs = class_splits[i]
        X_chunk = X.loc[idxs]
        y_chunk = y.loc[idxs]
        X_chunk.to_csv(f'dataset/{ip_nodes[i]}/x_{prefix}.csv', index=False, header=False)
        y_chunk.to_csv(f'dataset/{ip_nodes[i]}/y_{prefix}.csv', index=False, header=False)
        print(f"[NON-IID] Part {i+1} saved as {prefix}_{ip_nodes[i]}")


warnings.filterwarnings('ignore')

# Load and preprocess dataset
df_0 = pd.read_csv('dataset/dataset.csv')
df = df_0.copy()

X = df.drop(["Label"], axis=1)
y = df["Label"]

le_grouped = preprocessing.LabelEncoder()
y = pd.Series(le_grouped.fit_transform(y), index=df.index)
y = pd.get_dummies(y, dtype=float)

# Conteggio e visualizzazione delle classi target
print("Numero di classi target:", y.nunique())
print("\nDistribuzione delle classi (codici numerici):")
print(y.value_counts())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)

# Scale the features, but retain the original indices
scaler = MinMaxScaler()
X_train_index = X_train.index
X_test_index = X_test.index
X_columns = X.columns

# Applica lo scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ricrea i DataFrame con indici e colonne
X_train = pd.DataFrame(X_train, columns=X_columns, index=X_train_index)
X_test = pd.DataFrame(X_test, columns=X_columns, index=X_test_index)

# Sampling 5% of the data
sample_indices_train = X_train.sample(frac=1, random_state=42).index
X_train_sample = X_train.loc[sample_indices_train]
y_train_sample = y_train.loc[sample_indices_train]

sample_indices_test = X_test.sample(frac=1, random_state=42).index
X_test_sample = X_test.loc[sample_indices_test]
y_test_sample = y_test.loc[sample_indices_test]

# Save the sampled data to CSV files
X_train_sample.to_csv("dataset/X_train.csv", index=False, header=False)
X_test_sample.to_csv("dataset/X_test.csv", index=False, header=False)
y_train_sample.to_csv("dataset/y_train.csv", index=False, header=False)
y_test_sample.to_csv("dataset/y_test.csv", index=False, header=False)

split_and_save_csv_non_iid(X_train_sample, y_train_sample, n_nodes, prefix='train')
split_and_save_csv(X_test_sample, y_test_sample, n_nodes, prefix='test')

#split_and_save_csv_non_iid(X_train_sample, y_train_sample, n_nodes, prefix='train')
#split_and_save_csv(X_test_sample, y_test_sample, n_nodes, prefix='test')
