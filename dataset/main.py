# Import Librerie
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main():
    # Load NSL-KDD dataset
    train = pd.read_csv("KDDTrain+.txt", header=None, sep=',')
    test = pd.read_csv("KDDTest+.txt", header=None, sep=',')

    columns = (['duration'
        , 'protocol_type'
        , 'service'
        , 'flag'
        , 'src_bytes'
        , 'dst_bytes'
        , 'land'
        , 'wrong_fragment'
        , 'urgent'
        , 'hot'
        , 'num_failed_logins'
        , 'logged_in'
        , 'num_compromised'
        , 'root_shell'
        , 'su_attempted'
        , 'num_root'
        , 'num_file_creations'
        , 'num_shells'
        , 'num_access_files'
        , 'num_outbound_cmds'
        , 'is_host_login'
        , 'is_guest_login'
        , 'count'
        , 'srv_count'
        , 'serror_rate'
        , 'srv_serror_rate'
        , 'rerror_rate'
        , 'srv_rerror_rate'
        , 'same_srv_rate'
        , 'diff_srv_rate'
        , 'srv_diff_host_rate'
        , 'dst_host_count'
        , 'dst_host_srv_count'
        , 'dst_host_same_srv_rate'
        , 'dst_host_diff_srv_rate'
        , 'dst_host_same_src_port_rate'
        , 'dst_host_srv_diff_host_rate'
        , 'dst_host_serror_rate'
        , 'dst_host_srv_serror_rate'
        , 'dst_host_rerror_rate'
        , 'dst_host_srv_rerror_rate'
        , 'attack'
        , 'level'])

    train.columns = columns
    test.columns = columns

    train.attack.map(lambda a: 0 if a == 'normal' else 1).to_csv("train_targhet.csv", header=False, index=False)
    test.attack.map(lambda a: 0 if a == 'normal' else 1).to_csv("test_targhet.csv", header=False, index=False)
    # pd.get_dummies(train['attack'], columns=['attack']).to_csv("train_targhet.csv", header=False, index=False)
    # pd.get_dummies(test['attack'], columns=['attack']).to_csv("test_targhet.csv", header=False, index=False)
    train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])

    one_hot_encoding_colum = ['protocol_type', 'service', 'flag']
    numerical_columns = [col for col in columns if col not in one_hot_encoding_colum and col not in ['attack']]

    train = pd.get_dummies(train, columns=one_hot_encoding_colum)
    test = pd.get_dummies(test, columns=one_hot_encoding_colum)

    one_hot_columns_train = [col for col in train.columns if any(c in col for c in one_hot_encoding_colum)]
    train[one_hot_columns_train] = train[one_hot_columns_train].astype(int)
    one_hot_columns_test = [col for col in test.columns if any(c in col for c in one_hot_encoding_colum)]
    test[one_hot_columns_test] = test[one_hot_columns_test].astype(int)

    scaler = MinMaxScaler()
    train[numerical_columns] = scaler.fit_transform(train[numerical_columns])
    test[numerical_columns] = scaler.fit_transform(test[numerical_columns])

    train.to_csv("train_input.csv", header=False, index=False)
    test.to_csv("test_input.csv", header=False, index=False)



if __name__ == "__main__":
    main()
