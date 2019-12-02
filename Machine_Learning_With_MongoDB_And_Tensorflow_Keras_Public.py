#Basic imports
import itertools
import bson


#ML Packages:
#Tensorflow - using Keras as abstraction
import tensorflow as tf

#Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

#Data-Preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


#Model Evaluation / Accuracy
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import auc, confusion_matrix


#Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns


#Database
from pymongo import MongoClient
import pickle


#Business Understanding - we want to predict prices of an Air-BnB more accuratly to allow us to only buy highly profitable real estate

#Data Understanding
#Using MongoDB Compass
#Weekly and Monthly price are directly linked to the daily price - thus it is not effective to use them as variables
#What is more interesting is to look at factors, that give an indication for the quality of the flat.
#The customer satisfaction is hard to measure, as number of reviews can be good or bad - and we do not have a sentiment for the comments
#Therefore number of amenities and their kind (has different datastructures - as we are only storing the ones that are available - no null values in MongoDB - every document can be different in structure)
#the hight of the cleaningfee
#Cancellation policy and theirlike are deemed to be interesting

#Lets fetch some data for Barcelona:
#Ensure, that the Cluster has the sample dataset imported - and replace your username and connections string
client = MongoClient("MongoDB-Connection-String")
db = client.sample_airbnb
collection = db.listingsAndReviews


db_ml = client.ml
collection_reg_tree = db_ml.regression_tree_training
collection_tf = db_ml.neural_net


base_data = collection.aggregate([
    {
        '$geoNear': {
            'near': {
                'type': 'Point',
                'coordinates': [
                    2.159422869438373, 41.4061813436721
                ]
            },
            'maxDistance': 10000,
            'distanceField': 'distance',
            'spherical': True
        }
    }, {
        '$match': {
            'first_review': {
                '$exists': True
            },
            'bathrooms': {
                '$lte': 3
            },
            'bedrooms': {
                '$lte': 4
            },
            'review_scores.review_scores_cleanliness': {
                '$exists': True
            },
            'review_scores.review_scores_location': {
                '$exists': True
            }
        }
    }, {
        '$project': {
            'amenities': 1,
            'bathrooms': 1,
            'bedrooms': 1,
            'cancellation_policy': 1,
            'cleaning_fee': 1,
            'superhost': '$host.host_is_superhost',
            'minimum_nights': 1,
            'room_type': 1,
            'security_deposit': 1,
            'price': 1
        }
    }
])

base_data_list = list(base_data)
base_data_df = pd.DataFrame(base_data_list)

#Turn amenities into one hot encoded features using sklearn
mlb = MultiLabelBinarizer()
one_hot_encoded_features = pd.DataFrame(mlb.fit_transform(base_data_df["amenities"]),
                                           columns=mlb.classes_,
                                           index=base_data_df.index)

filtered_labels = [label for label in one_hot_encoded_features.columns if "translation" not in label]
one_hot_encoded_features = one_hot_encoded_features[filtered_labels]

#choose only labels that do not allow for individual link
one_hot_encoded_features_selected = one_hot_encoded_features.loc[:,(one_hot_encoded_features.sum()>one_hot_encoded_features.shape[0]*0.1)]

#Now turn all other categorial Variables into Dummies
room_type = pd.get_dummies(base_data_df["room_type"],prefix=["room_type"])
cancellation_policy = pd.get_dummies(base_data_df["cancellation_policy"],prefix=["cancellation_policy"])

#Drop List Column and join one hot encoded columns based on PK
base_data_df.drop(columns="amenities",inplace=True)
base_data_df.drop(columns="room_type",inplace=True)
base_data_df.drop(columns="cancellation_policy",inplace=True)
base_data_df.drop(columns="_id",inplace=True)

base_data_df = base_data_df.join(other=one_hot_encoded_features_selected)
base_data_df = base_data_df.join(other=room_type)
base_data_df = base_data_df.join(other=cancellation_policy)

#Type Conversion
base_data_df =  base_data_df.apply(lambda x: x.astype(str).astype(float) if isinstance(x.iloc[0], bson.Decimal128) else x)


#Start cleaning the data
base_data_df.isnull().sum().plot.bar()
plt.show()
#Lets restore some data that deemed to be lost
#A non existent Security Deposit can be expected to be an equivalent to 0
#A non existent Cleaning Fee can be expected to be an equivalent of 0
#There are no further NULL values:
base_data_df = base_data_df.fillna(value=0)

#Removing Outliers
non_binary_columns = [column for column in base_data_df if np.isin(base_data_df[column],[0,1],invert=True).sum()>0]
sns.pairplot(base_data_df[non_binary_columns])
plt.show()

#Remove Security Deposit > 1500
base_data_df = base_data_df.loc[base_data_df["security_deposit"]<1500]
#Remove Price > 750
base_data_df = base_data_df.loc[base_data_df["price"]<750]

#Remove Cleaning Fee
base_data_df = base_data_df.loc[base_data_df["cleaning_fee"]<200]

#Review data
sns.pairplot(base_data_df[non_binary_columns])
plt.show()


#Define Features and Target Variable
target_variable = base_data_df["price"]
features = base_data_df.drop(columns = "price")


#Split into training and test dataset
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    target_variable,
                                                    test_size=0.10,
                                                    random_state=42)

features_column_names = features.columns
target_variable_name = "price"

scaler = MinMaxScaler()
scaler = scaler.fit(x_train)
x_train_scaled = pd.DataFrame(scaler.transform(x_train),columns=features_column_names)
x_test_scaled = pd.DataFrame(scaler.transform(x_test),columns=features_column_names)
x_train_scaled_array = scaler.transform(x_train)
x_test_scaled_array = scaler.transform(x_test)




#Do Feature Seclection
#Deep Learning will reduce weights to 0 in training process and additional dropout will increase the regularization effect / simplification and Random Forrests have inherent Regularization by having a limited depth -> Automatically only selects most relevant features
#For simplification of the training process of deep learning models one can leverage feature importance of the way simpler to train regression tree (mainly sensitive to linear dependecies and thus overestimating continous variables)
#Keeping in mind the sensitivity - it is necessary to artificially overweigh some of the categorial variables
# Import packages for NN / Decision Tree

# RandomRegressionTree
kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 7)
results = []
for depth in range(1,7,1):
    for min_samples in range(10,50,10):
        run = 1
        for train,test in kf.split(x_train_scaled,y_train):
            regression_tree = RandomForestRegressor(max_depth=depth, min_samples_leaf=min_samples)
            regression_tree.fit(x_train_scaled.iloc[train],y_train.iloc[train])
            score = regression_tree.score(x_train_scaled.iloc[test],y_train.iloc[test]) #Returns the coefficient of determination R^2 of the prediction
            collection_reg_tree.insert_one({"hyperparameters_id": f"{depth},{min_samples}",
                                            "depth": depth, "min_samples": min_samples,
                                            "run": run,
                                            "score": score,
                                            "model": pickle.dumps(regression_tree)})
        run += 1


train_results = collection_reg_tree.aggregate([
                                            {
                                                '$group': {
                                                    '_id': '$hyperparameters_id',
                                                    'average_score': {
                                                        '$avg': '$score'
                                                    },
                                                    'depth': {
                                                        '$first': '$depth'
                                                    },
                                                    'min_samples': {
                                                        '$first': '$min_samples'
                                                    },
                                                    'scores': {
                                                        '$push': '$score'
                                                    },
                                                    'models': {
                                                        '$push': '$model'
                                                    }
                                                }
                                            }, {
                                                '$sort': {
                                                    'average_score': -1,
                                                    'depth': 1,
                                                    'min_samples': -1
                                                }
                                            }
                                        ])

train_results = list(train_results)
train_results_df = pd.DataFrame(train_results)
winning_model_params = train_results_df.iloc[1]

regression_tree = RandomForestRegressor(max_depth=winning_model_params["depth"],
                                        min_samples_leaf=winning_model_params["min_samples"])
winning_model = regression_tree.fit(x_train_scaled,
                                    y_train)

#Which are most important features
feature_importances_df = pd.DataFrame(columns=features_column_names)
feature_importances = winning_model.feature_importances_
feature_importances_df.loc[0] = feature_importances
most_important_features = feature_importances_df.T.sort_values(0,
                                                                ascending=False).iloc[0:15].index


#Build Model / Use MDB for storing training information
#Using dropout to simplify the overall model and add further regularization
x_train_scaled_selected = x_train_scaled[most_important_features]
x_test_scaled_selected = x_test_scaled[most_important_features]

for neurons_layer_1 in [5,10,15,20]:
    for neurons_layer_2 in [2,5,10]:
        for dropout_layer_1 in [0,0.2]:
            for dropout_layer_2 in [0,0.2]:
                for epochs in [200,400]:
                    for activation_function in ["relu"]:
                        model = tf.keras.Sequential()
                        model.add(tf.keras.layers.Dense(neurons_layer_1,input_dim = 15, activation=activation_function))
                        model.add(tf.keras.layers.Dropout(dropout_layer_1))
                        model.add(tf.keras.layers.Dense(neurons_layer_2, activation = activation_function))
                        model.add(tf.keras.layers.Dropout(dropout_layer_2))
                        model.add(tf.keras.layers.Dense(1,activation="linear"))

                        model.compile(loss = "mean_squared_logarithmic_error", optimizer = "adam", metrics=["mse"])

                        history = model.fit(x_train_scaled_selected,y_train,epochs = epochs, batch_size= 100 ,validation_split = 0.15)

                        collection_tf.insert_one({
                                                  "epochs": epochs,
                                                  "val_mse_final": [float(x) for x in history.history["val_mse"]][-1],
                                                  "val_loss_final": [float(x) for x in history.history["val_loss"]][-1],
                                                  "neurons_layer_1":neurons_layer_1,
                                                  "neurons_layer_2":neurons_layer_2,
                                                  "dropout_layer_1":dropout_layer_1,
                                                  "dropout_layer_2":dropout_layer_2,
                                                  "activation_function":activation_function,
                                                  "identifier":f"{neurons_layer_1},"
                                                               f"{neurons_layer_2},"
                                                               f"{activation_function}",
                                                 "history": {
                                                     "loss": [float(x) for x in history.history["loss"]],
                                                     "val_loss": [float(x) for x in history.history["val_loss"]],
                                                     "mse": [float(x) for x in history.history["mse"]],
                                                     "val_mse": [float(x) for x in history.history["val_mse"]]
                                                    },
                                                  "model": model.to_json()
                                                  }
                                                 )


best_nns = collection_tf.aggregate(
                                    [
                                        {
                                            '$sort': {
                                                'val_mse_final': 1,
                                                'val_loss_final': 1
                                            }
                                        }, {
                                            '$limit': 25
                                        }
                                    ]
                                    )
best_nns = list(best_nns)
best_nns_df = pd.DataFrame(best_nns)
best_nn_model = best_nns_df.iloc[0]

best_model = tf.keras.models.model_from_json(best_nn_model["model"])
best_model.compile(loss = "mean_squared_logarithmic_error",
                   optimizer = "adam",
                   metrics=["mse"])

history = best_model.fit(x_train_scaled_selected,
                        y_train,
                        epochs = best_nn_model["epochs"],
                        batch_size= 100 ,
                        validation_split = 0.15)


#Summarize history for loss
fig, ax = plt.subplots(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Accuracy during training
fig, ax = plt.subplots(figsize=(10,5))
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predicted_values = best_model.predict(x_test_scaled_selected)
predicted_values = [value[0] for value in predicted_values]

comparison = pd.DataFrame({"prices": y_test, "predicted_prices":predicted_values})

fig, ax = plt.subplots(figsize=(10,5))
comparison.plot.bar(rot=0)
plt.show()


#Huray we have build aa model with extremly diverse data - very efficiently and natively


#Cleanup
collection_reg_tree.drop()
collection_tf.drop()