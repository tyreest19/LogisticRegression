
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def costFunction(x, y, m, theta):
    #return np.transpose(y) * np.log(sigmoid(x * theta)) + np.transpose(1 - y) * np.log(1 - sigmoid(x * theta))
    loss = 0
    for i in range(m):
        z = np.dot(
            np.transpose(theta),
            x[i]
        )
        loss += y[i] * np.log(sigmoid(z)) + (1 - y[i]) * np.log(1 - sigmoid(z))
    return -(1/m) * loss

def gradientDescent(x, y, m, theta, alpha, iterations=1500):
    #return theta - (alpha/m)* np.transpose(x) * (sigmoid(x * theta) - y)
    for iteration in range(iterations):
        for j in range(len(theta)):
            gradient = 0
            for i in range(m):
                z = np.dot(
                    np.transpose(theta),
                    x[i]
                )
                gradient += (sigmoid(z) - y[i]) * x[i][j]
            theta[j] = theta[j] - ((alpha/m) * gradient)
        print('Current Error is:', costFunction(x, y, m, theta))
    return theta

def test(x, y, m, theta):
    correct = 0
    for i in range(m):
        z = np.dot(
                np.transpose(theta),
                x[i]
            )
        predicted_value = sigmoid(z)
        if predicted_value >= 0.5 and y[i] == 1:
            correct += 1
        elif predicted_value < 0.5 and y[i] == 0:
            correct += 1
    return correct/m, (1 - (correct/m))

def updateTypeColumn(dataframe, columnName, columnValue):
    for index, row in dataframe.iterrows():
        if row.Type_1 == columnValue:
            dataframe.loc[index, columnName] = 1
        else:
            dataframe.loc[index, columnName] = 0

if __name__ == '__main__':

    data = pd.read_csv('data/pokemon_alopez247.csv')
    data = data.drop(['Type_2'], axis=1)
    data.dropna()
    updateTypeColumn(data, 'Type_1', 'Fire')
    data['Type_1'] = data['Type_1'].apply(int)
    print(data.corr())
    X = [[row[1]['Sp_Atk'], row[1]['Pr_Male']] for row in    
          data.iterrows()]
    y = [row[1]['Type_1'] for row in data.iterrows()]
    training_features, testing_features, training_output, testing_output = train_test_split(X, 
                   y, 
                   test_size=0.7,  
                   train_size=0.3,    
                   random_state=42)
    theta = np.random.uniform(size=len(training_features[0]))
    print('Final theta\'s \n', 
            gradientDescent(training_features, 
            training_output, 
            len(training_features[0]), theta, 0.001))
    accuracy_rate, error_rate = test(testing_features, testing_output, len(testing_output), theta)
    print('Accuracy: {accuracy} \nError: '
      '{error}'.format(accuracy=accuracy_rate,           
       error=error_rate))
