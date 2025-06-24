import pandas as pd
import numpy as np
import warnings
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

## Configuration parameters
n_nodes = 5

# Create the folder for each node
base_ip = "172.19.0."
ip_nodes = [f"{base_ip}{i+2}" for i in range(n_nodes)]
for i in range(n_nodes):
    os.mkdir(f'dataset/{ip_nodes[i]}')


def split_and_save_csv(X, y, n_parts, prefix):
    base_chunk_size = len(X) // n_parts
    remainder = len(X) % n_parts
    start_idx = 0
    for i in range(n_parts):
        chunk_size = base_chunk_size + 1 if i < remainder else base_chunk_size
        end_idx = start_idx + chunk_size
        X_chunk = X.iloc[start_idx:end_idx]
        y_chunk = y.iloc[start_idx:end_idx]
        X_chunk.to_csv(f'dataset/{ip_nodes[i]}/{prefix}_x_train.csv', index=False, header=False)
        y_chunk.to_csv(f'dataset/{ip_nodes[i]}/{prefix}_y_train.csv', index=False, header=False)
        print(f"Part {i+1} saved as {prefix}_{ip_nodes[i]}_training.csv")
        
        start_idx = end_idx

warnings.filterwarnings('ignore')

# Load and preprocess dataset
df_0 = pd.read_csv('dataset/KDDTrain.txt')
df = df_0.copy()

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]
df.columns = columns
df = df.dropna()

# Binary classification: "normal" vs "attack"
df['attack'] = df['attack'].apply(lambda x: "normal" if x == 'normal' else "attack")

# Label encoding for categorical columns
cat_features = df.select_dtypes(include='object').columns
le = preprocessing.LabelEncoder()
clm = ['protocol_type', 'service', 'flag', 'attack']
for x in clm:
    df[x] = le.fit_transform(df[x])

X = df.drop(["attack"], axis=1)
y = df["attack"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)

# Retain the original indices for X_train and X_test
X_train_indices = X_train.index
X_test_indices = X_test.index

# Select a subset of columns for training
columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'wrong_fragment', 
           'hot', 'logged_in', 'num_compromised', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
           'rerror_rate']
X_train = X_train[columns]
X_test = X_test[columns]

# Scale the features, but retain the original indices
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled data back to DataFrame with the original indices
X_train = pd.DataFrame(X_train_scaled, columns=columns, index=X_train_indices)
X_test = pd.DataFrame(X_test_scaled, columns=columns, index=X_test_indices)

# Sampling 5% of the data
sample_indices_train = X_train.sample(frac=0.5, random_state=42).index
X_train_sample = X_train.loc[sample_indices_train]
y_train_sample = y_train.loc[sample_indices_train]

sample_indices_test = X_test.sample(frac=0.5, random_state=42).index
X_test_sample = X_test.loc[sample_indices_test]
y_test_sample = y_test.loc[sample_indices_test]

# Save the sampled data to CSV files
X_train_sample.to_csv("dataset/X_train.csv", index=False, header=False)
X_test_sample.to_csv("dataset/X_test.csv", index=False, header=False)
y_train_sample.to_csv("dataset/y_train.csv", index=False, header=False)
y_test_sample.to_csv("dataset/y_test.csv", index=False, header=False)

split_and_save_csv(X_train_sample, y_train_sample, n_nodes, prefix='partition')
split_and_save_csv(X_test_sample, y_test_sample, n_nodes, prefix='partition_test')