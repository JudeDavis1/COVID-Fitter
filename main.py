import sys
import csv
import numpy as np

def load_csv():
    final_list = []
    final_list_x = []
    final_list_y = []

    f = open("covid_data.csv")
    reader = csv.DictReader(f)

    for row in reader:
        final_list.append(int(row["newCasesByPublishDate"]))
    
    midpoint = int(len(final_list) / 2)
    
    final_list_x.append(
        final_list[:midpoint]
    )
    final_list_y.append(
        final_list[midpoint:]
    )
    
    return np.array(
        [
            np.array(final_list_x),
            np.array(final_list_y)
        ]
    )


try:
    print("Loading libraries")

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from regression import LogisticRegression

    print("Initializing Regression model")

    model = LogisticRegression()

    x, y = load_csv()

    print("Training model")
    model.fit(x, y)  # fit the model

    prediction = model.predict(x)  # overall prediction
    plt.scatter(x, y)

    print(f"Model Prediction: {prediction}\n\n")
    print("Model Accuracy: " + str(model.accuracy) + "\n\n")

    if prediction >= .5:
        print("A second wave of COVID-19 is quite likely")
    else:
        print("It is likely that there will NOT be a second wave")
except KeyboardInterrupt:
    sys.exit()