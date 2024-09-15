#Random Forest Classifier for Breast Cancer Detection
Overview
This project implements a Random Forest Classifier to predict whether a tumor is malignant or benign using the Breast Cancer dataset from sklearn. The project includes training the model, making predictions, visualizing the results, and performing hyperparameter tuning to optimize the model.

Project Structure
Data Loading: Breast cancer dataset is loaded using load_breast_cancer() from sklearn.datasets.
Train-Test Split: The dataset is split into training and testing sets using train_test_split with stratification to maintain class distribution.
Model Training: A Random Forest Classifier (RandomForestClassifier) is trained using the training data.
Model Prediction: Predictions are made on the test data.
Evaluation: The performance of the model is evaluated using the classification_report from sklearn.metrics and mislabeling statistics.
Tree Export: The first three decision trees from the random forest are exported and visualized using graphviz.
Visualization: The decision boundary of the Random Forest model is visualized for two selected features from the training set.
Hyperparameter Tuning: Grid search is performed using GridSearchCV to find the best hyperparameters for the model.
Requirements
To run this project, you need the following libraries:

pandas
numpy
matplotlib
scikit-learn
graphviz
Install the necessary libraries using:

bash
Copy code
pip install pandas numpy matplotlib scikit-learn graphviz
How to Run the Code
Load the Dataset: The load_breast_cancer() function is used to load the breast cancer dataset, which contains 30 features related to tumor characteristics.
Split the Data: The data is split into training and testing sets (70% train, 30% test).
Train the Model: A RandomForestClassifier is trained on the training data.
Make Predictions: The trained model is used to make predictions on the test set.
Evaluate the Model: The performance is evaluated using a classification report, and the number of mislabeled points is calculated.
Export Trees: The first three decision trees from the Random Forest are visualized using graphviz.
Visualize Results: A 2D plot of the decision boundary for the first two features is created to visualize the classification regions.
Hyperparameter Tuning: A grid search is performed to find the best hyperparameters for the model using GridSearchCV.
Results
The classifier's performance is reported using precision, recall, F1-score, and support for each class (malignant/benign).
Mislabeling statistics are provided to show the number of incorrect predictions.
Visualization of the decision boundary helps in understanding how the model classifies the data.
Hyperparameter tuning using GridSearchCV outputs the best parameters and the corresponding performance score.
Example Output
Sample output from the classification report:

plaintext
Copy code
              precision    recall  f1-score   support
           0       0.96      0.93      0.95       108
           1       0.96      0.97      0.96       180
    accuracy                           0.96       288
   macro avg       0.96      0.95      0.96       288
weighted avg       0.96      0.96      0.96       288
Hyperparameter Tuning
Best Parameters after Grid Search:

plaintext
Copy code
n_estimators: 300
max_depth: 8
criterion: 'gini'
Future Work
Explore feature engineering techniques to improve the model's performance.
Experiment with other classifiers such as Gradient Boosting, XGBoost, or AdaBoost.
Implement cross-validation with more advanced techniques.
