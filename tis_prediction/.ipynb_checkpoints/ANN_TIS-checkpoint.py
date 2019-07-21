#Load dataset
# Creating a DataFrame of given iris dataset.
# Load the Pandas libraries with alias 'pd'
from sklearn.model_selection import cross_val_predict, cross_val_score
import pandas as pd
# Read data from file 'tis_data.csv'
# (in the same directory that your python process is based)
datatis = pd.read_csv("tis_data.csv")
# Preview the first 5 lines of the loaded data
datatis.head()

#X=datatis[['PJ','SS','KG','PR','PH','STT','DTT','TE','PC']]  # Features
X=datatis[['PJ','SS','PH','STT','DTT','TE','PC']]  # Features without KG & PR
Y=datatis['Real']  # Labels


from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(activation= 'relu', random_state= 8, hidden_layer_sizes= (50),
                    learning_rate_init= 0.05, learning_rate= 'invscaling', momentum= 0.1, max_iter= 50, verbose= 0)
# activation= relu & logistic & tanh & identity, relu: y=0 (x<0) & y=x (x=>0), hidden_layer_sizes: number of neurons in each layer
# learning_rate_init: initial learning rate (<0.2), learning_rate: constant or decreasing learning rate
# verbose: see training process
scores = cross_val_score(MLP, X, Y, cv= 10, scoring= 'f1')
#scores.mean()

#y_pred=MLP.predict(X_test)
