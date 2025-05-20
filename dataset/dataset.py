import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# import kagglehub
# path = kagglehub.dataset_download("oddrationale/mnist-in-csv")

path = "C:\\Users\\play_\\CLionProjects\\automatic_aifes\\dataset\\"

# Lettura del dataset
df = pd.read_csv(path + "mnist_train.csv")

df = df[df["label"].isin([0, 1, 2])]
df = (
    df[df["label"].isin([0, 1, 2])]
    .groupby("label")
    .apply(lambda x: x.sample(n=5000, random_state=42))
    .reset_index(drop=True)
)

train_list = []
test_list = []

for label in [0, 1, 2]:
    class_df = df[df["label"] == label]
    train = class_df.sample(frac=0.8, random_state=42)
    test = class_df.drop(train.index)
    train_list.append(train)
    test_list.append(test)

# Concatenazione finale
train_df = pd.concat(train_list).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_list).sample(frac=1, random_state=42).reset_index(drop=True)


X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))

X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

y_train_df = pd.DataFrame(y_train_encoded, columns=encoder.get_feature_names_out(["label"]))
y_test_df = pd.DataFrame(y_test_encoded, columns=encoder.get_feature_names_out(["label"]))

# Salvataggio su file CSV
X_train_df.to_csv(path+"x_train.csv", index=False, header=False)
y_train_df.to_csv(path+"y_train.csv", index=False, header=False)
X_test_df.to_csv(path+"x_test.csv", index=False, header=False)
y_test_df.to_csv(path+"y_test.csv", index=False, header=False)

print(X_train_df.shape)
print(X_train_df.size)

print(y_train_df.shape)
print(y_train_df.size)

print(X_test_df.shape)
print(X_test_df.size)

print(y_test_df.shape)
print(y_test_df.size)