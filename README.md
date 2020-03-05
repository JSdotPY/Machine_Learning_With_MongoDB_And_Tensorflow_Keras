# AirBnB Rental Price Prediction - Machine Learning With MongoDB And Tensorflow Keras
Example on how to conduct efficient Machine-Learning with Python, MongoDB, Tensorflow (Keras).

Python has become the "Swiss Army Knife" for Developers and Datascientists all around the world.
By programming in Python nearly all of the most recent technologies can be used as native connectors and APIs are provided.
  - MongoDB
  - Tensorflow
  - Spark
  - ...

The code in this repository is one of the foundations for my technical talks on ML.
It showcases how MongoDB can be leveraged jointly with Keras that has been incorporated in Tensorflow.

As within my Demo on MongoDB with Mongolite and R, I think that using a Database, that is natural to work with is extremly important for Developer (my own) productivity.

The sample shows how:
- How the Aggregation Pipeline can be used to prepare base tables from polymorphic data
- Models can be stored in MongoDB
- How the Aggregation Pipeline can be used to select the best model across different training runs

All of this allows to reduce the workload on the machine used for the training processes and even more importantly allows for continous training and deployment architectures.
Especially when using ensemble methods, this comes in handy.

# The Slide-Deck
[Working Draft](https://docs.google.com/presentation/d/1Ny-xHH4DnpYZRJEuM2zXeP43gRS9nwtO6uFBYI_Cg5k/edit?usp=sharing)

# The Demo
The Demo can be found in the Machine_Learning_With_MongoDB_And_Tensorflow_Keras_Public.py file.

### Installation
- Go to http://atlas.mongodb.com and create a free Account.
- Create a Cluster (M0 is free)
- Click on the three dots next to the connect button on your cluster and choose load sample dataset <br>

Once you have done that and the cluster has finished the initialization / update:<br>
Please adjust the connection-string to MongoDB Atlas and ensure to have loaded the sample datasets
```sh
    #MongoDB
    client = MongoClient("MongoDB-Connection-String")
```
Please also add your current IP-Adress to the IP-Whitelist under the security section on Atlas - otherwise you will not be able to connect to the cluster as the VPC used to deploy the cluster is fully locked down.
Now you can run the Demo.

### Packages

| Package | Link |
| ------ | ------ |
| PyMongo | [https://api.mongodb.com/python/current/] |
| Tensorflow | [https://www.tensorflow.org/] |
| SkLearn | [https://scikit-learn.org/stable/] |
| Matplotlib | [https://matplotlib.org/] |
| Pandas | [https://pandas.pydata.org/] |
| Numpy | [https://numpy.org/] |
| Seaborn | [https://seaborn.pydata.org/] |
| Pickle (Might need to be replaced) | [https://docs.python.org/3/library/pickle.html] |




This is a manual comment:
Markdown Created with: https://dillinger.io/
