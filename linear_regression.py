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

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def denormalize(y_norm, min_val, max_val):
    return y_norm * (max_val - min_val) + min_val

if __name__ == "__main__":
    theta0, theta1 = 0, 0
    try:
        theta0, theta1 = load_model()
        x, y = load_data('data.csv')
        mileage = float(input("Entrez le kilométrage de la voiture : "))
        mileage = normalize(mileage, np.min(x), np.max(x))
        # denormaliser le prix
        mileagen = denormalize(mileage, np.min(x), np.max(x))
        price = estimate_price(mileage, theta0, theta1)
        pricen = denormalize(price, np.min(y), np.max(y))
        print(f"Prix estimé de la voiture : {pricen:.2f}")
        plt.figure()
        plt.scatter(x, y, color='blue', label='Données normalized')
        # Tracer la ligne de régression denormalisés
        predicted_y = theta0 + theta1 * normalize(x, np.min(x), np.max(x))
        plt.plot(x, denormalize(predicted_y, np.min(y), np.max(y)), color='red', label='Ligne de régression')
        # Tracer une croix rouge pour le prix estimé
        plt.scatter(mileagen, pricen, color='green', label='Prix estimé')
        # Labels et titre
        plt.xlabel('Kilométrage (km)')
        plt.ylabel('Prix (€)')
        plt.title('Données - Kilométrage vs Prix normalized')
        plt.legend()
        plt.show()
        
    except FileNotFoundError:
        print("Erreur : Le modèle n'a pas encore été entraîné. Théta0 et Théta1 sont set à 0.")
        mileage = float(input("Entrez le kilométrage de la voiture : "))
        price = estimate_price(mileage, theta0, theta1)
        print(f"Prix estimé de la voiture : {price:.2f}")