# Noisy Exponential Activation Function, NEAF
This repository contains a research project I performed. The project is focused around the implementation of a custom 
activation function I created called the Noisy Exponential Activation Function, NEAF. NEAF injects noise into the network and maps the data to a set range. Unlike previous noise injecting activation functions, NEAF injects noise dampened over an exponential function allowing for the activation function to serve as a regressor and also preserve the non-linear attributes desired from an activation function. In this repository, you can find an implementation of NEAF on a Deep Recurrent Attention Unit built from scratch using Numpy example for Time Series forecasting for stock data. You can also find an individual implementation of NEAF and a pdf for the mathematical deviations for a Recurrent Attention Unit. 


### Prerequisites
Python3: numpy, pandas, sklearn, matplotlib, seaborn

### Installing
```
 pip install numpy pandas sklearn matplotlib seaborn
```

## Running the Network

### Import Iris data, split, and scale

The network is tested with the Iris data set. The training samples and test samples, X, need to be a numpy array with the shape of (sample, features). The labels of the training samples and test samples, Y, need the class labels to be assigned to natural numbers (1, 2, 3, 4, .....) and also be in the form of a numpy array. The values of X need to be scaled to the interval of (0,1).

```
    # --- Import Iris data, split, and scale --- #
    data = pd.read_csv('iris_data_norm.csv')
    data = data.iloc[:,1:]
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
    X = scaler_min_max.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    y_train, y_test = y_train.values, y_test.values
    X_train, X_test = X_train.T, X_test.T  
```

### Declare network
To use the network, it needs to be declared by calling the class "ReflexFuzzyNeuroNetwork" from the file GRMMFN.py. Two tuning parameters are passed during the declaration, gamma and theta. Gamma serves as a tuning factor for the sensitivity of the membership function and theta serves a tuning factor for the expansion criteria. Ranges for these values most often fall within the interval [2,4] for gamma and [.1,1] for theta.

```
    # --- Declare network --- #
    nn = ReflexFuzzyNeuroNetwork(gamma=2, theta=.1)
```

### Train and Test Network
The X inputs and Y labels need to be separated into two separate date sets, a training and test set. The X-values
are the first passed parameter for "train" and "test" functions and, the y-values are the second parameter passed for the 
"train" and "test" functions. 

```
    # --- Train network --- #
    nn.train(X_train, y_train)

    # --- Test Network --- #
    nn.test(X_test,y_test)
```

The program finishes by printing out a graph of the networks fit to training data and forecast on test data.
```
![Image of forecast graph]
(GraphImage.png)
```


## Authors

* **Enrique Nueve** 

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* Guoqiang Zhong, Guohua Yue, Xiao Ling for creating the Recurrent Attention Unit.
