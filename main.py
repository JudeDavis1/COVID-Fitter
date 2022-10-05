import sys
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

from regression import LogisticRegression

BATCH = 100

def main():
    try:
        print("[*] Initializing Regression model")

        model = LogisticRegression(epochs=100, lr=0.01)
        model.activation = model._relu

        (x, y) = load_csv()

        train_x = x / np.max(x)
        train_y = y / np.max(y)
        

        print("[*] Training model")
        model.fit(train_x, train_y)  # fit the model

        plt.scatter(x, y)

        i = random.randint(1, 250)

        prediction = model.predict(train_x)  # overall prediction

        print(prediction[i] * np.max(y))
        print(y[i])

        plt.plot(x, prediction * np.max(y))
        plt.show()

        print(f"[*] Model Accuracy: {model.accuracy:.3f}\n\n")
    except KeyboardInterrupt:
        sys.exit()

def load_csv() -> Tuple[np.ndarray, int]:
    X = []
    Y = []

    f = open("covid_data.csv")
    reader = csv.DictReader(f)

    for row in reader:
        Y.insert(0, int(row["newCasesByPublishDate"]))
    
    X = np.array(range(len(Y)))
    Y = np.array(Y)

    ret = [
        X,
        Y
    ]

    return ret


if __name__ == '__main__':
    main()