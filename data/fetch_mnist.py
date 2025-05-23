from sklearn.datasets import fetch_openml
import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np


mnist_path = os.path.join(os.path.dirname(__file__), "datasets", "mnist.csv")

class MNIST_Fetcher:
    """
    A class to fetch the MNIST dataset.
    """
    def __init__(self):
        self.mnist = None
        self.features = None
        self.labels = None

    def get_data(self):
        self.mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        self.features, self.labels = self.mnist['data'], self.mnist['target']
        self.labels = self.labels.astype(np.float64)
        self.features = self.features / 255.0
        return self.features, self.labels

    def get_train_data(self):
        return self.features[:60000], self.labels[:60000]

    def get_test_data(self):
        return self.features[60000:], self.labels[60000:]
    
    def save_data(self):
        with open(mnist_path, 'wb') as f:
            pickle.dump(self.mnist, f)

    def load_data(self):
        with open(mnist_path, 'rb') as f:
            self.mnist = pickle.load(f)
        self.features, self.labels = self.mnist['data'], self.mnist['target']
    
    def draw_10_images(self):
        plt.figure(figsize=(10, 10))
        for i in range(100):
            plt.subplot(10, 10, i + 1)
            plt.imshow(self.features[i].reshape(28, 28), cmap='gray')
            plt.title(f"Label: {self.labels[i]}", fontsize=8)
            plt.axis('off')  # 隐藏坐标轴
        plt.subplots_adjust(wspace=0.1, hspace=0.3)  # 减小水平和垂直间距
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.show()

    def draw_10_images_with_predicted_label(self, predicted_labels):
        plt.figure(figsize=(10, 10))
        for i in range(100):
            plt.subplot(10, 10, i + 1)
            plt.imshow(self.features[i].reshape(28, 28), cmap='gray')
            plt.title(f"Predicted: {predicted_labels[i]};", fontsize=8)
            plt.axis('off')


# if __name__ == "__main__":
#     mnist_fetcher = MNIST_Fetcher()
#     mnist_fetcher.get_data()
#     mnist_fetcher.save_data()
