import argparse
from data.fetch_mnist import MNIST_Fetcher, mnist_path
from data.fetch_uci import UCI_fetcher
from data.fetch_credit import CreditFetcher
import os
"""
Execute the script with the following command at the root directory of the project:
python model_test/<script_name>.py --name <dataset_name>
"""
def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='linear model test')
    parser.add_argument('--name', type=str, help='dataset_name')
    # if parse_args.name == "uci":
    #     parse_args.name = ''
    return parser.parse_args()

"""
Select the dataset based on the name
"""
def dataset_selecter(name: str):
    if name == "mnist":
        mnist_fetcher = MNIST_Fetcher()
        if os.path.exists(mnist_path):
            mnist_fetcher.load_data()
        else:
            mnist_fetcher.get_data()
            mnist_fetcher.save_data()
        train_X, train_y = mnist_fetcher.get_train_data()
        test_X, test_y = mnist_fetcher.get_test_data()
    elif name == "uci":
        uci_fetcher = UCI_fetcher()
        train_X, test_X, train_y, test_y = uci_fetcher.load_data()
    elif name == "credit":
        credit_fetcher = CreditFetcher()
        train_X, test_X, train_y, test_y = credit_fetcher.load_data(encoding='ohe')
    return train_X, test_X, train_y, test_y
