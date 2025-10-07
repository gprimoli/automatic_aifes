import pandas as pd
import numpy as np
import warnings
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold

## Configuration parameters
n_nodes = 5
base_ip = "172.19.0."
ip_nodes = [f"{base_ip}{i+2}" for i in range(n_nodes)]
for i in range(n_nodes):
    try:
        os.mkdir(f'dataset/{ip_nodes[i]}')
    except:
        print("Folder already created.")

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
# X_train_sample.to_csv("dataset/X_train.csv", index=False, header=False)
# X_test_sample.to_csv("dataset/X_test.csv", index=False, header=False)
# y_train_sample.to_csv("dataset/y_train.csv", index=False, header=False)
# y_test_sample.to_csv("dataset/y_test.csv", index=False, header=False)

# split_and_save_csv(X_train_sample, y_train_sample, n_nodes, prefix='train')
# split_and_save_csv(X_test_sample, y_test_sample, n_nodes, prefix='test')

#split_and_save_csv_non_iid(X_train_sample, y_train_sample, n_nodes, prefix='train')
#split_and_save_csv(X_test_sample, y_test_sample, n_nodes, prefix='test')

import os
import numpy as np
import pandas as pd

def dirichlet_non_iid_partitions(y_onehot: pd.DataFrame, n_parts: int, alpha: float = 0.3, seed: int = 42):
    """
    Crea n_parts partizioni non-IID per label-skew usando una distribuzione Dirichlet(α) per classe.
    - y_onehot: DataFrame one-hot delle etichette (righe = campioni)
    - n_parts:  numero di nodi
    - alpha:    parametro Dirichlet (più piccolo = più non-IID)
    Ritorna: lista di array di indici, uno per nodo.
    """
    rng = np.random.RandomState(seed)
    y_labels = y_onehot.idxmax(axis=1).to_numpy()
    classes = np.unique(y_labels)

    # Indici per classe
    idx_by_class = {c: np.where(y_labels == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    # Costruisci le quote per ogni classe ~ Dirichlet
    parts = [[] for _ in range(n_parts)]
    for c in classes:
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue
        props = rng.dirichlet([alpha] * n_parts)                 # quota per nodo
        counts = (props * len(idxs)).astype(int)

        # aggiusta per eventuale arrotondamento
        while counts.sum() < len(idxs):
            counts[rng.randint(0, n_parts)] += 1

        start = 0
        for i in range(n_parts):
            end = start + counts[i]
            if end > start:
                parts[i].extend(idxs[start:end].tolist())
            start = end

    # Shuffle finale per nodo
    for i in range(n_parts):
        parts[i] = np.array(rng.permutation(parts[i]), dtype=int)
    return parts


def save_train_parts_per_node(X: pd.DataFrame, y_onehot: pd.DataFrame, parts, ip_nodes, prefix='train'):
    """Salva x_train/y_train per nodo + un file di riepilogo classi."""
    y_labels_all = y_onehot.idxmax(axis=1)
    for i, idx in enumerate(parts):
        node_dir = f'dataset/{ip_nodes[i]}'
        os.makedirs(node_dir, exist_ok=True)
        X_chunk = X.iloc[idx]
        y_chunk = y_onehot.iloc[idx]
        X_chunk.to_csv(f'{node_dir}/x_{prefix}.csv', index=False, header=False)
        y_chunk.to_csv(f'{node_dir}/y_{prefix}.csv', index=False, header=False)

        # log distribuzione
        y_lab = y_labels_all.iloc[idx]
        dist = y_lab.value_counts().to_dict()
        with open(f'{node_dir}/split_info_{prefix}.txt', 'w') as f:
            f.write(f"[DIRICHLET α={alpha}] node {ip_nodes[i]} | n={len(idx)}\n")
            for lab, cnt in sorted(dist.items()):
                f.write(f"class {lab}: {cnt}\n")
        print(f"[DIRICHLET] saved {prefix} for {ip_nodes[i]} (n={len(idx)})")


def replicate_test_to_all_nodes(X_test: pd.DataFrame, y_test: pd.DataFrame, ip_nodes, prefix='test'):
    """Replica lo stesso set di test su tutti i nodi (valutazione coerente)."""
    for ip in ip_nodes:
        node_dir = f'dataset/{ip}'
        os.makedirs(node_dir, exist_ok=True)
        X_test.to_csv(f'{node_dir}/x_{prefix}.csv', index=False, header=False)
        y_test.to_csv(f'{node_dir}/y_{prefix}.csv', index=False, header=False)
    print(f"[TEST] replicated to {len(ip_nodes)} nodes.")

# ========= dopo la tua preparazione di X_train, X_test, y_train, y_test =========

# (opzionale) se volevi davvero un sotto-campione, imposta frac=0.05 (ora era 1.0)
sample_frac = 1.0
X_train_sample = X_train.sample(frac=sample_frac, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]
X_test_sample  = X_test   # tieni il test completo
y_test_sample  = y_test

# IID stratificato del TRAIN in n nodi
#parts = stratified_iid_partitions(X_train_sample, y_train_sample, n_nodes, seed=42)
#save_train_parts_per_node(X_train_sample, y_train_sample, parts, ip_nodes, prefix='train')

# Parametri non-IID
alpha = 9   # più piccolo => più non-IID (es. 0.1 molto sbilanciato; 10 ~ quasi IID)
seed  = 42

# 1) genera le partizioni
parts = dirichlet_non_iid_partitions(y_train_sample, n_nodes, alpha=alpha, seed=seed)

# (opzionale) controllo: nessun nodo vuoto?
for i, p in enumerate(parts):
    if len(p) == 0:
        # Se capita con dataset molto piccolo, redistribuisci 1 campione a quel nodo
        # (edge case raro con α molto basso)
        donor = max(range(n_nodes), key=lambda j: len(parts[j]))
        parts[i] = np.array([parts[donor][0]], dtype=int)
        parts[donor] = parts[donor][1:]

save_train_parts_per_node(X_train_sample, y_train_sample, parts, ip_nodes, prefix='train')

replicate_test_to_all_nodes(X_test_sample, y_test_sample, ip_nodes, prefix='test')

