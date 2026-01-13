# Assignments

## Module 1 Assignment: Cyber ML Model Experiments

**Goal**: Step into the role of a cybersecurity data scientist.

Your assignment is to step into the role of a cybersecurity data scientist. You will apply the techniques from the lab to a new slice of the data, making and justifying your own feature engineering decisions to build the most effective classifier you can.

*   **Task 1**: Reload the full CICIDS2017 DDoS dataset, not the 2,000-sample subset used in the lab.
*   **Task 2**: Perform the same data cleaning steps: replace `inf` values and drop any rows with `NaN` values.
*   **Task 3**: Select a **different** pair of two features from the dataset than the ones we used in the lab (' Fwd Packet Length Mean' and ' Flow Duration'). (Note: Be mindful of the leading spaces in the column names when selecting your features.) In a text block in your notebook, write a brief justification for why you chose these two features.
*   **Task 4**: Create a new training and testing split using your chosen features and the corresponding labels.
*   **Task 5**: Train two separate models on your new data: a `KNeighborsClassifier` and a `SVC` (Support Vector Classifier). You must use a `StandardScaler` to scale the data for both models. For the `SVC`, use the best hyperparameters found during the lab (C=10, gamma=0.01).
*   **Task 6**: For each of your two trained models, generate and display the `classification_report` and a `confusion_matrix` plot.
*   **Task 7**: In a final text block, write a short paragraph comparing the performance of your kNN and SVM models on your chosen features. Which model performed better? Why do you think that might be? Use the precision, recall, and F1-score from your classification reports to support your conclusion.

## Planned Assignments

*(Assignments for later modules will be added here)*

1. EDA and Parameter Optimization
2. Clustering Exploration Lab
3. TensorFlow Playground
4. Phishing Email Detection
