import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename)
    return data['km'].values, data['price'].values

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def denormalize(y_norm, min_val, max_val):
    return y_norm * (max_val - min_val) + min_val

def train_linear_regression(x, y, learning_rate=0.01, epochs=100000):
    # fonction de cout
    m = len(y)
    theta0, theta1 = 0, 0  # Initialisation des paramètres
    
    #gradient descent algorithm
    for _ in range(epochs):
        predictions = theta0 + theta1 * x
        error = predictions - y
        
        tmp_theta0 = theta0 - learning_rate * (1/m) * np.sum(error) # je crois que ça je n'ai pas le droit
        tmp_theta1 = theta1 - learning_rate * (1/m) * np.sum(error * x)
        
        theta0, theta1 = tmp_theta0, tmp_theta1
    
    return theta0, theta1

def save_model(theta0, theta1):
    np.save('model.npy', [theta0, theta1])

def plot_data_and_line(x, y, theta0, theta1, xn, yn):
    
    # Tracer la chart du set de data
    plt.figure()
    plt.scatter(xn, yn, color='blue', label='Données réelles')
    plt.title("Set de Data")
    plt.legend()


    # Tracer la ligne de régression + chart normalized
    plt.figure()
    plt.scatter(x, y, color='blue', label='Données normalized')
    predicted_y = theta0 + theta1 * x
    plt.plot(x, predicted_y, color='red', label='Ligne de régression')
    # Labels et titre
    plt.xlabel('Kilométrage (km)')
    plt.ylabel('Prix (€)')
    plt.title('Données - Kilométrage vs Prix normalized')

    # Afficher la légende
    plt.legend()
    # Afficher la chart
    plt.show()

if __name__ == "__main__":    

    x, y = load_data('data.csv')
    xtmp = x
    ytmp = y
    min_max_y = np.min(y), np.max(y)
    min_max_x = np.min(x), np.max(x)
    x = normalize(x)  # Normalisation des valeurs de mileage
    y = normalize(y)
    
    theta0, theta1 = train_linear_regression(x, y)
    save_model(theta0, theta1)
    plot_data_and_line(x, y, theta0, theta1, xtmp, ytmp)
    
    print(f"Modèle entraîné : theta0 = {theta0}, theta1 = {theta1}")
