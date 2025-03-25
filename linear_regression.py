import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename):
    data = pd.read_csv(filename)
    return data['km'].values, data['price'].values

def load_model():
    return np.load('model.npy')

def estimate_price(mileage, theta0, theta1):
    return (theta0 + (theta1 * mileage))

if __name__ == "__main__":
    theta0, theta1 = 0, 0
    try:
        theta0, theta1 = load_model()
        x, y = load_data('data.csv')
        mileage = float(input("Entrez le kilométrage de la voiture : "))
        #mileage = (mileage - np.min(x)) / (np.max(x) - np.min(y))
        price = estimate_price(mileage, theta0, theta1)
        print(f"Prix estimé de la voiture : {price:.2f}")
        
    except FileNotFoundError:
        print("Erreur : Le modèle n'a pas encore été entraîné. Théta0 et Théta1 sont set à 0.")
        mileage = float(input("Entrez le kilométrage de la voiture : "))
        price = estimate_price(mileage, theta0, theta1)
        print(f"Prix estimé de la voiture : {price:.2f}")