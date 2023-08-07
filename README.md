# deep-learning-challenge
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures.

## Overview
This repository creates a binary classification model to predict whether an organisation funded by the fictional company ‘Alphabet Soup’ will be successful by training the model on a variety of features within the provided dataset (CSV file covering more than 34,000 organisations).

### Step 1: Preprocess the Data
1) Read in the charity_data.csv file to a Pandas DataFrame.
2) Identify the target variable(s) and the feature(s) for your model.
3) Drop the EIN and NAME columns from the dataset.
4) Determine the number of unique values for each column.
5) For columns with more than 1 unique values, bin "rare" categorical variables together in a new value, Other.
6) Use pd.get_dummies() to encode categorical variables.
7) Split the preprocessed data into features array (X) and target array (y).
8) Split the data into training and testing datasets using the train_test_split function.
9) Scale the training and testing features datasets using scikit-learn's StandardScaler.

### Step 2: Compile, Train, and Evaluate the Model
1) Using TensorFlow and Keras, design a neural network model for binary classification.
2) Determine the number of input features and nodes for each layer in your model.
3) Create the first hidden layer with an appropriate activation function.
4) Add a second hidden layer with an appropriate activation function if necessary.
5) Create an output layer with an appropriate activation function for binary classification.
6) Compile and train the model, and evaluate its loss and accuracy using the test data.
7) Create a callback to save the model's weights every five epochs.
8) Save the trained model to an HDF5 file named "AlphabetSoupCharity.h5".

### Step 3: Optimise the Model
1) Create a new Jupyter Notebook named "AlphabetSoupCharity_Optimisation.ipynb".
2) Import the necessary dependencies and read in the charity_data.csv to a Pandas DataFrame.
3) Preprocess the dataset as in Step 1, and make any necessary adjustments that came out of optimising the model. (Target accuracy of 75%)
4) Save and export the optimised model's weights to an HDF5 file named "AlphabetSoupCharity_Optimisation.h5".

## Dataset
The dataset used for this project is available in the "charity_data.csv" file. The CSV file contains information about organisations that received funding from the Alphabet Soup foundation, along with various features and metadata about each organisation. A link to this file can be found in the .ipynb files.

## Summary of results
The model initially had a positive result with an accuracy of 73.03%, and after multiple attempts of changing values, layers and methods, the highest accuracy achieved was 73.08% which can be found in the AlphabetSoupCharity_Optimisation file. Unfortunately it was not possible for the model to achieve the 75% target with the columns used and another model would potentially be able to achieve the target accuracy. For a further breakdown, please refer to the attached PDF file within the repository called "AlphabetSoupAnalysisReport.pdf".

## Requirements
To run the code in this repository, you'll need the following dependencies:
* Pandas
* NumPy
* TensorFlow
* Scikit-learn

## Instructions for Running the Code
Clone this repository to your local machine.
Make sure you have all the required dependencies installed in your environment.
Run the Jupyter Notebook files in order, starting with "AlphabetSoupCharity.ipynb" for data preprocessing, model training, and evaluation, and then "AlphabetSoupCharity_Optimisation.ipynb" for model optimisations that were undertaken.
