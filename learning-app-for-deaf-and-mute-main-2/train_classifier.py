import pickle  # Import the pickle module for loading and saving data
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier from sklearn for classification
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting the dataset
from sklearn.metrics import accuracy_score  # Import accuracy_score for evaluating the model
import numpy as np  # Import numpy for array manipulation

# Load the processed data and labels from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert the data and labels from the dictionary to numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the dataset into training and testing sets
# 80% of the data will be used for training, and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Predict the labels for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model's predictions
score = accuracy_score(y_predict, y_test)

# Print the accuracy score as a percentage
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a pickle file for later use
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
