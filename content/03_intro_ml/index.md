# Module 3: Intro to ML

## 3.0 Foundations of AI and Machine Learning

### 3.1 Demystifying Artificial Intelligence

The term "Artificial Intelligence" is often used as a broad and sometimes vague catch-all. Its definition seems to shift as technology progresses. To build a solid foundation for this course, it is crucial to establish a clear, structured framework for understanding AI and its key sub-disciplines.

Two pioneers in the field captured this fluid definition perfectly:

> "AI is whatever hasn't been done yet." — Douglas Hofstadter

> "As soon as it works, no one calls it AI anymore." — John McCarthy

To bring clarity to this, we can think of these concepts as a set of nested hierarchies, moving from the general to the specific:

*   **Artificial Intelligence (AI)**: This is the broadest concept, encompassing any technique that enables computers to emulate human-like learning, reasoning, and problem-solving.
    *   **Machine Learning (ML)**: A subset of AI, machine learning focuses specifically on algorithms that learn patterns and make predictions directly from data, without being explicitly programmed with rules.
        *   **Deep Learning (DL)**: A subset of machine learning that utilizes complex, multi-layered neural networks to process data and learn from it. This approach is particularly effective for complex tasks involving large datasets, such as image recognition and natural language processing.
            *   **Generative AI**: A subset of deep learning focused on models that can generate new, original content—such as text, images, or code—based on the patterns they have learned from training data.

Now that we have defined the terminology, let's explore the primary methodologies that machine learning models use to learn from data.

### 3.2 Core Methodologies of Machine Learning

Machine learning is not a single, monolithic technique. It is a collection of different approaches, each suited to different types of data and different kinds of problems. In cybersecurity, we primarily draw from three core methodologies.

**Supervised Learning**

Supervised learning is the most common form of machine learning. It relies on **labeled data**, where each data point is tagged with a known correct output or "answer." The model learns by finding the patterns that map the input data to the correct output label. Its two primary tasks are:

*   **Classification**: The model learns to assign an input to a predefined category. For example, classifying an email as "Spam" or "Not Spam," or identifying network traffic as "Benign" or "DDoS."
*   **Regression**: The model learns to predict a continuous numerical value. For example, predicting the financial risk score of a potential data breach or forecasting the number of malicious login attempts a system will face.

**Unsupervised Learning**

In contrast to supervised learning, unsupervised learning works with **unlabeled data**. The goal is not to predict a known answer but to discover hidden patterns, structures, and relationships within the data itself.

*   The primary use case in cybersecurity is **anomaly detection**. The model is trained on a dataset of "normal" system or network behavior. It learns the baseline patterns and can then flag any new activity that significantly deviates from this learned norm, signaling a potential threat.
*   Another key task is **clustering**, where the model groups similar data points together without any prior knowledge of what those groups represent.

**Reinforcement Learning**

Reinforcement learning is a different paradigm inspired by behavioral psychology. It involves an "**agent**" that learns to make decisions by performing actions within an "**environment**" to maximize a cumulative "**reward**."

*   A simple analogy is teaching a dog to sit. When the dog performs the correct action (sitting), it receives a reward (a treat), reinforcing that behavior.
*   The process works as a continuous feedback loop: the agent **Observes** the state of the environment, takes an **Action**, receives a **Reward** or **Penalty**, and **Adjusts** its strategy to achieve better rewards in the future.

These foundational concepts are the building blocks for the models we will now construct in our hands-on lab.

---

## 3.3 Hands-On Lab: Building Your First Cybersecurity ML Models

### 3.3.1 Lab Setup and Prerequisites

This section marks our transition from theory to practice. We will now walk through the process of building, training, and evaluating several fundamental machine learning models using Python, scikit-learn, and a Google Colab notebook. This is where the concepts we've discussed become tangible skills. By the end of this lab, you will have built and evaluated multiple models, providing a solid foundation for every project that follows.

While this course does not require prior ML knowledge, it is assumed that you have a basic familiarity with algebra (linear equations), statistics (mean, histograms), and some Python programming. The Colab notebook for this lab contains links to refresher materials if you need them.

*   **Task**: Please download the `Fall_2025_CYBERSEC_520_Class_1.ipynb` file (linked below).

A Colab notebook consists of two main components: **Text blocks**, like this one, which provide explanations, and **Code blocks**, which contain executable Python code. It is critical that you run the code blocks in the order they appear to ensure that variables and models are defined correctly.

### 3.3.2 Part 1: Linear Regression and the Concept of Loss

Our first model will be a simple linear regression. The primary goal here is not just to build the model, but to develop an intuitive understanding of a core machine learning concept that applies to nearly every model we will build: the **loss function**.

**Step 1: Generate and Plot Data**

Run the first code cell under the heading "It starts with Data." This code will generate a set of data points that follow a linear trend but include some random "noise," simulating a real-world dataset. It will then create a scatter plot of this data.

**Step 2: Understand the Goal**

Our objective is to find the single straight line—the "line of best fit"—that best represents the relationship between the X and Y variables in our data.

**Step 3: Introduce the Loss Function**

How do we mathematically define the "best" fit? We use a metric called a **loss function**. For linear regression, the most common loss function is the **Mean Squared Error (MSE)**. The MSE calculates the average of the squared differences between the actual data points and the predicted values on our line. By squaring the errors, it heavily penalizes predictions that are far off from the actual values, making it sensitive to outliers. The entire goal of the "learning" process is to find the parameters for our line (its slope and intercept) that result in the lowest possible MSE.

**Step 4: Let the Machine Learn**

Run the code cells under the heading "Let the Machine 'Learn' From Data." This code uses the LinearRegression model from the scikit-learn library. The model algorithmically finds the optimal slope and intercept that minimize the MSE for our dataset. The notebook will then print the final results and plot the line of best fit.

### 3.3.3 Part 2: Classification with k-Nearest Neighbors (kNN)

Now we will move from regression (predicting a number) to classification (predicting a category). We will use the classic Palmer Penguins dataset to build a model that can classify a penguin's species (Adelie, Chinstrap, or Gentoo) based on its physical measurements like bill and flipper length.

**Step 1: Load and Clean the Data**

Run the cells to load the penguins dataset from a URL. We then use the `.dropna()` function to remove any rows that have missing data, which is a common first step in data preparation. The output of `penguins.species_short.value_counts()` shows us the distribution of the three species in our dataset.

**Step 2: Split Data for Training and Testing**

This is a critical concept in machine learning. We must split our data into two parts: a **training set**, which the model uses to learn patterns, and a **testing set**, which is held back and used to evaluate the model's performance on data it has never seen before. This practice is essential to prevent **overfitting**, a scenario where a model memorizes the training data but fails to generalize to new, unseen data. Run the `train_test_split` code cell to perform this split.

**Step 3: Train a kNN Model**

The k-Nearest Neighbors (kNN) algorithm is one of the simplest classification methods. To classify a new, unlabeled data point, it looks at the 'k' closest labeled data points (its "neighbors") in the training set and assigns the new point the majority class of those neighbors. Run the cells that import `KNeighborsClassifier`, create an instance of the model, and `.fit()` it to the training data. The initial accuracy on the test data is approximately 78.6%.

**Step 4: Introduce Hyperparameters and Data Scaling**

**Hyperparameters** are settings that we, the developers, choose to control the learning process itself. They are not learned from the data. For kNN, the number of neighbors (k) is a key hyperparameter.

**Data scaling** is a crucial preprocessing step for distance-based algorithms like kNN. Our dataset has features with very different scales (e.g., body mass in grams is a large number, while culmen depth in millimeters is small). Without scaling, the features with large scales would unfairly dominate the distance calculation. Scaling puts both features on a level playing field so that both contribute meaningfully to the distance calculation. We use a `StandardScaler` to transform all features to a similar scale. Run the cell that applies this scaler.

**Step 5: Visualize the Effect of Scaling**

To see the impact of scaling, run the code that generates a UMAP visualization. UMAP is a technique that reduces the dimensionality of our data so we can plot it in 2D. As you can see in the plots, scaling the data results in much cleaner and more distinct clusters for each penguin species. This visual separation suggests that our model will be able to perform much better.

**Step 6: Retrain and Evaluate**

Run the final `make_pipeline` code cell. A pipeline is a convenient way to chain together multiple steps, in this case, the scaler and the kNN classifier. After fitting this pipeline to the training data, the new accuracy score on the test data is a perfect 1.0 (or 100%), demonstrating the dramatic impact of proper data scaling.

**Step 7: Visualize the Decision Boundary**

Finally, run the code that generates a `plot_decision_regions` graph. This plot shows us exactly how the trained kNN model has partitioned the feature space. Each colored region represents the area where any new data point would be classified as a particular species.

### 3.3.4 Part 3: A Real-World Cyber Example - DDoS Detection

Now, let's apply these classification techniques to a real-world cybersecurity problem. In this section, we will use the CICIDS2017 dataset, which contains records of real network traffic, to build a classifier that can distinguish between normal (BENIGN) traffic and a Distributed Denial of Service (DDoS) attack.

**Step 1: Load, Clean, and Subsample the Data**

Run the cells to load the DDoS dataset. As with the penguins data, the first step is cleaning. We replace any infinite (inf) values and drop rows with missing (NaN) values. Because the full dataset is very large, we will select a random subsample of 2,000 data points to allow for faster, more efficient analysis and visualization in this lab.

**Step 2: Train the DDoS Classifier**

Run the cells that select two features for our model ('Fwd Packet Length Mean' and 'Flow Duration'), split the data into training and testing sets, and then train a kNN classifier pipeline. The resulting accuracy is approximately 89.6%.

**Step 3: Introduce Advanced Evaluation Metrics**

In cybersecurity, **accuracy** can be a dangerously misleading metric. Imagine a dataset where 99.99% of traffic is benign and only 0.01% is an attack. A model could achieve 99.99% accuracy by simply labeling everything as "benign," yet it would fail to catch a single attack. We need more nuanced metrics. The **Confusion Matrix** helps us break down performance:

*   **True Positives (TP)**: An attack that is correctly identified as an attack. (Good!)
*   **True Negatives (TN)**: Benign traffic that is correctly identified as benign. (Good!)
*   **False Positives (FP)**: Benign traffic incorrectly flagged as an attack. (Bad - causes alert fatigue).
*   **False Negatives (FN)**: An actual attack that the model missed. (Very Bad - the most dangerous error).

From these, we derive more informative metrics:

*   **Precision (TP / (TP + FP))**: Answers the question, "Of all the alerts generated, what percentage were actual attacks?" High precision means the model is reliable and doesn't cry wolf.
    *   Prioritize Precision when: The cost of a false positive is high. For example, automatically blocking a legitimate C-suite executive's account based on a false alert would be highly disruptive.
*   **Recall (TP / (TP + FN))**: Answers the question, "Of all the real attacks that occurred, what percentage did our model catch?" High recall means the model is comprehensive and doesn't miss threats.
    *   Prioritize Recall when: The cost of a false negative is catastrophic. For example, in malware detection, you would rather have a few false alarms than miss a single instance of ransomware.
*   **F1 Score**: The harmonic mean of precision and recall. It provides a single score that balances the trade-off between the two, which is extremely useful for comparing models, especially on imbalanced datasets.

Run the `classification_report` cell to see these metrics for our model and the cell that plots the confusion matrix. The resulting matrix for our kNN model shows 15 False Negatives (actual DDoS attacks that we missed) and 37 False Positives (benign traffic that was incorrectly flagged as an attack). Our goal with more advanced models is to drive both of these numbers down.

### 3.3.5 Part 4: A More Advanced Model - Support Vector Machines (SVM)

Next, we'll explore a more powerful classification model: the Support Vector Machine (SVM). The core idea behind an SVM is to find an optimal dividing line, or "hyperplane," that creates the largest possible margin or gap between different classes of data points.

For data that can't be separated by a straight line, SVMs use a powerful technique called the **Kernel Trick**. Imagine your data points are scattered on a flat sheet of paper. The kernel trick allows the SVM to warp or bend that paper into a third dimension, making it possible to slice through the data with a simple flat plane (the hyperplane) that cleanly separates the classes. The Radial Basis Function (RBF) kernel is a common method for achieving this.

**Step 1: Tune Hyperparameters with Grid Search**

Like kNN, SVMs have important hyperparameters that control their behavior, such as C (the regularization parameter) and gamma (the kernel coefficient). Finding the best combination of these can be a tedious manual process. Instead, we can automate it using **GridSearchCV**. This tool systematically tests a grid of specified parameter combinations and uses cross-validation to identify the set that yields the best performance. Run the `GridSearchCV` code cell. The search will identify the optimal parameters: {'C': 10, 'gamma': 0.01}.

**Step 2: Train and Evaluate the Final SVM Model**

Finally, run the last code cell. This creates a new SVM pipeline using our scaled data and the optimal hyperparameters we just found. After training and evaluating this new model, you will see a final weighted F1-score of 0.93, a noticeable improvement over the kNN model's score of 0.90. This demonstrates how a more advanced model combined with systematic hyperparameter tuning can lead to better performance.
