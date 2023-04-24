import pandas as pd
import matplotlib.pyplot as plt

def show_plot():
    Y_test = pd.read_csv('processed_data/Y_test.csv')
    Y_pred = pd.read_csv('processed_data/Y_pred.csv')

    plt.plot(Y_test, label='Actual')
    plt.plot(Y_pred, label='Predicted')

    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.xlim(100, 200)
    plt.show()