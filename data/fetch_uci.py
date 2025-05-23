from ucimlrepo import fetch_ucirepo 
import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

uci_path = os.path.join(os.path.dirname(__file__), "datasets", "uci_data")

class UCI_fetcher:
    def __init__(self):
        self.uci = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_data(self):
        self.uci = fetch_ucirepo(id=144).data
        features = pd.DataFrame(self.uci.features)
        labels = pd.DataFrame(self.uci.targets)
        features.columns = ["account_status","period","history_credit","credit_purpose","credit_limit",
               "saving_account","person_employee","income_installment_rate","marry_sex","other_debtor",
               "address","property","age","installment_plans","housing",
               "credits_num","job","dependents","have_phone","foreign_worker"]
        labels.columns = ["target"]
        labels.target = labels.target - 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    def save_data(self):
        self.X_train.to_csv(os.path.join(uci_path, "X_train.csv"), index=False)
        self.X_test.to_csv(os.path.join(uci_path, "X_test.csv"), index=False)
        self.y_train.to_csv(os.path.join(uci_path, "y_train.csv"), index=False)
        self.y_test.to_csv(os.path.join(uci_path, "y_test.csv"), index=False)

    def load_data(self):
        X_train_df = pd.read_csv(os.path.join(uci_path, "X_train.csv"))
        X_test_df = pd.read_csv(os.path.join(uci_path, "X_test.csv"))
        y_train_df = pd.read_csv(os.path.join(uci_path, "y_train.csv"))
        y_test_df = pd.read_csv(os.path.join(uci_path, "y_test.csv"))
        y_train = y_train_df.iloc[:, -1].values
        y_test = y_test_df.iloc[:, -1].values
        X_train_enc = pd.get_dummies(X_train_df)
        X_test_enc = pd.get_dummies(X_test_df)
        X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
        self.X_train = X_train_enc.values
        self.X_test = X_test_enc.values
        self.y_train = y_train
        self.y_test = y_test
        return self.X_train, self.X_test, self.y_train, self.y_test
    
# if __name__ == "__main__":
#     uci_fetcher = UCI_fetcher()
#     uci_fetcher.get_data()
#     uci_fetcher.save_data()
#     print(uci_fetcher.load_data())
