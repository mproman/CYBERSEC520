---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# CYBERSEC 520: Foundations of AI in Cybersecurity
## Class 2: Data Processing, Visualization, and Model Foundations

**Lecture Date:** Fall 2025  
**Topic:** Moving from raw network logs to robust predictive models.

---

### Agenda
1. **Recap:** Interactive Visualization of SVMs and Hyperparameters.
2. **Operational Realities:** Why NetFlow? Privacy, Encryption, and Scale.
3. **Data Processing:** Handling Infinity, Imputation Strategies, and "Dirty" Cyber Data.
4. **Visualization:** Unlocking high-dimensional data (PCA vs. t-SNE vs. UMAP).
5. **Modeling:** Naive Bayes, Decision Trees, Random Forests, XGBoost.
6. **Optimization:** Automated Hyperparameter Tuning.
7. **Activity:** In-class optimization challenge.

```{code-cell} ipython3
# --- LECTURE SETUP ---
# Run this cell first to ensure all plots look professional and readable on a projector.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import svm, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Suppress messy warnings for the lecture
warnings.filterwarnings('ignore')

# VISUALIZATION SETTINGS
# 'talk' context makes labels and lines larger for presentation slides
sns.set_context("talk") 
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14

print("Environment Setup Complete. Visualizations configured for Lecture Mode.")
```

## 1. Recap: Interactive Concepts

Last week we discussed how models learn boundaries and why hyperparameters matter. Instead of just defining them, let's **see** them.

### A. Visualizing the Decision Boundary (SVM)
The Support Vector Machine (SVM) tries to maximize the margin between classes. Two key hyperparameters control this:
* **C (Regularization):** How much do we care about misclassifying a few points? (High C = Strict, Low C = Loose).
* **Gamma (Kernel Coefficient):** How far does the influence of a single training example reach? (High Gamma = Close reach/Islands, Low Gamma = Far reach/Smooth).

```{code-cell} ipython3
# Generate sample data (Moons)
np.random.seed(0)
X_viz, y_viz = datasets.make_moons(200, noise=0.15)

# Scale the data
scaler = StandardScaler()
X_viz = scaler.fit_transform(X_viz)

# Create a mesh to plot in
x_min, x_max = X_viz[:, 0].min() - 0.5, X_viz[:, 0].max() + 0.5
y_min, y_max = X_viz[:, 1].min() - 0.5, X_viz[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Define hyperparameter ranges to visualize
C_range = [0.1, 1, 10]
gamma_range = [0.1, 1, 10]

# Create subplots
fig, axes = plt.subplots(len(C_range), len(gamma_range), figsize=(18, 15))
fig.suptitle("Recap: SVM Decision Boundaries (C vs Gamma)", fontsize=20)

for i, C in enumerate(C_range):
    for j, gamma in enumerate(gamma_range):
        # Train SVM
        clf = svm.SVC(C=C, gamma=gamma)
        clf.fit(X_viz, y_viz)

        # Plot decision boundary
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plotting
        axes[i, j].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
        axes[i, j].scatter(X_viz[:, 0], X_viz[:, 1], c=y_viz, cmap=plt.cm.RdYlBu, edgecolors='black', s=40)
        axes[i, j].set_title(f'C = {C}, gamma = {gamma}')
        axes[i, j].set_xticks(()) # Hide ticks for cleanliness
        axes[i, j].set_yticks(())

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
```

### B. Visualizing the "Search" (Grid Search)
When we talk about "Tuning a Model," we are essentially navigating a surface to find the peak performance.

Below is a heatmap of a **Grid Search**. We try every combination of `C` and `Gamma` to find the highest accuracy (bright yellow).
* **Dark Blue regions:** Poor hyperparameters (Underfitting or Overfitting).
* **Bright Yellow regions:** The "Sweet Spot" we are looking for.

```{code-cell} ipython3
# Define parameter grid (Log space search)
param_grid = {'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)}

# Run Grid Search
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_viz, y_viz)

# Extract results for plotting
scores = grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['gamma']))

# Plot Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(scores, annot=True, fmt='.3f', cmap='viridis',
            xticklabels=[f'{x:.1e}' for x in param_grid['gamma']],
            yticklabels=[f'{x:.1e}' for x in param_grid['C']])

plt.title('Hyperparameter Optimization Surface (Grid Search Recap)')
plt.xlabel('Gamma')
plt.ylabel('C')

# Display best params
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_:.4f}")
plt.show()
```

## 2. Operational Realities: The Data Problem

Before we touch code, we must ask: **Are we collecting the right data?**

### The PCAP vs. NetFlow Debate
In a perfect world, we would use **Full Packet Capture (PCAP)**. It contains the payload, the files, everything.
**The Reality:**
* **Scale:** PCAP generates Terabytes of data per day. Processing this in real-time for ML is often impossible.
* **Encryption:** Most payloads are now encrypted (TLS 1.3). You can't see the bad stuff anyway without breaking SSL (Privacy/Compliance nightmare).

### The Solution: NetFlow / IPFIX
We use metadata. Who talked to whom? How long? How many bytes? What was the port?
* **Lightweight:** Fraction of the size of PCAP.
* **Privacy Preserving:** We don't see the user's passwords or emails, just the traffic behavior.

**Today's Dataset:** We are using a flow-based dataset (CIC-IDS) similar to NetFlow.

```{code-cell} ipython3
# --- INSTALL MISSING PACKAGES ---
# Uncomment if running in a fresh Colab environment
# !pip install umap-learn[plot] xgboost
```

```{code-cell} ipython3
# --- DATA LOADING ---

# OPTION 1: Real Data (Google Drive)
# from google.colab import drive
# drive.mount('/content/drive')
# file_path = 'drive/My Drive/CYBERSEC520/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
# df = pd.read_csv(file_path)

# OPTION 2: Synthetic Data (FOR LECTURE DEMO)
# This ensures the notebook runs immediately without needing the 500MB file.

def generate_synthetic_portscan_data(n_rows=5000):
    np.random.seed(42)
    
    # Generate Benign Traffic
    n_benign = int(n_rows * 0.8)
    benign_df = pd.DataFrame({
        'Flow Duration': np.random.exponential(1000, n_benign),
        'Total Fwd Packets': np.random.randint(1, 20, n_benign),
        'Destination Port': np.random.choice([80, 443, 22, 53, 8080], n_benign, p=[0.4, 0.4, 0.05, 0.1, 0.05]),
        'Packet Length Std': np.random.normal(50, 10, n_benign),
        'Flow Bytes/s': np.random.normal(5000, 2000, n_benign),
        'Label': 'BENIGN'
    })
    
    # Generate PortScan Traffic
    n_attack = n_rows - n_benign
    attack_df = pd.DataFrame({
        'Flow Duration': np.random.exponential(50, n_attack), # Scans are fast
        'Total Fwd Packets': np.random.randint(1, 3, n_attack), # Scans are small
        'Destination Port': np.random.randint(1, 65535, n_attack), # Scans hit random ports
        'Packet Length Std': np.random.normal(0, 1, n_attack), # Scans are uniform size
        'Flow Bytes/s': np.random.choice([np.inf, 1000000], n_attack), # Fast/abnormal flows
        'Label': 'PortScan'
    })
    
    # Combine
    df = pd.concat([benign_df, attack_df]).sample(frac=1).reset_index(drop=True)
    
    # Add some noise/NaNs to make it realistic for cleaning
    mask = np.random.choice([True, False], size=len(df), p=[0.01, 0.99])
    df.loc[mask, 'Flow Bytes/s'] = np.nan
    
    return df

# Load Data
try:
    # Try loading real data if you have it mounted
    df = pd.read_csv('drive/My Drive/CYBERSEC520/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
    print("Loaded REAL Dataset from Drive.")
except:
    # Fallback to synthetic
    print("Real dataset not found. Generating SYNTHETIC PortScan data for demonstration.")
    df = generate_synthetic_portscan_data()

print(f"Dataset Shape: {df.shape}")
df.head()
```

## 3. Visualization & Dimensionality Reduction

In cybersecurity, we often deal with "high-dimensional" data (dozens or hundreds of features). Visualizing this is impossible without reducing those dimensions down to 2 or 3.

We will compare three popular techniques to see how they group our PortScan attacks.

### 1. PCA (Principal Component Analysis)
* **What it is:** A linear technique that finds the "principal components" (directions) that maximize the variance.
* **Pros:** Fast, deterministic, preserves global structure.
* **Cons:** Fails to capture non-linear relationships (complex attack patterns).

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
* **What it is:** A non-linear probabilistic technique designed specifically for visualization.
* **Pros:** Excellent at revealing local clusters (grouping specific attack tools).
* **Cons:** Slow, and **distance between clusters is meaningless**.

### 3. UMAP (Uniform Manifold Approximation and Projection)
* **What it is:** A newer manifold learning technique.
* **Pros:** Faster than t-SNE, preserves both local clustering and global structure.
* **Cons:** Newer, requires parameter tuning.

```{code-cell} ipython3
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- 1. PREP DATA FOR VIZ ---
# We sample 2000 points because t-SNE/UMAP are computationally heavy
subset_size = 2000
df_sample = df.sample(n=min(subset_size, len(df)), random_state=42).copy()

# Temporary cleaning just for visualization (we will do robust cleaning later)
df_viz = df_sample.select_dtypes(include=[np.number]).fillna(0)
df_viz = df_viz.replace([np.inf, -np.inf], df_viz.replace([np.inf, -np.inf], np.nan).max().max())
y_viz = df_sample['Label']

# Scale the data (Critical for PCA/UMAP)
scaler = StandardScaler()
X_viz_scaled = scaler.fit_transform(df_viz)

# --- 2. RUN ALGORITHMS ---
print("Running PCA...")
pca = PCA(n_components=2)
res_pca = pca.fit_transform(X_viz_scaled)

print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
res_tsne = tsne.fit_transform(X_viz_scaled)

print("Running UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
res_umap = reducer.fit_transform(X_viz_scaled)

# --- 3. PLOT ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

def plot_viz(embedding, title, ax):
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y_viz, palette='viridis', s=60, alpha=0.7, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(title='Class')

plot_viz(res_pca, "PCA (Linear/Global)", axes[0])
plot_viz(res_tsne, "t-SNE (Local Clusters)", axes[1])
plot_viz(res_umap, "UMAP (Balanced)", axes[2])

plt.tight_layout()
plt.show()
```

## 4. Data Processing: The "Grunt Work"

### 4.1 Imputation Strategies
Missing data is common (sensor outages, timeouts). How we fill it matters.

* **Numerical Data (Mean/Median):** Good for general stats, but can mask outliers.
* **Time Series (Forward/Backward Fill):** Best for logs. If a sensor dies at 12:00 and 12:02, it was likely active at 12:01. We use the last known state.
* **Model-Based (KNN Imputation):** Finds similar rows (neighbors) and borrows their value. Most accurate but computationally expensive.

### 4.2 The Problem with Infinity
In network data (like CIC-IDS), columns like `Flow Bytes/s` are calculated as `Total Bytes / Duration`. 
If a packet flow happens instantaneously (Duration = 0), we get **Infinity**.

**Bad approach:** Replacing Inf with an arbitrary huge number (e.g., `1e10`).
* *Why?* Distance-based models (KNN, SVM) will see this as a massive outlier, crushing all other variance.

**Better approach:** 
1.  Flag the Infinity (Create `is_inf` column).
2.  Clamp the value to the maximum *observed finite* value.

```{code-cell} ipython3
# Work on a copy
df_clean = df.copy()

# Identify columns with Infinity
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
inf_cols = [col for col in numeric_cols if np.isinf(df_clean[col]).any()]

print(f"Columns with Infinity: {inf_cols}")

for col in inf_cols:
    # 1. Calculate max finite value
    max_val = df_clean.loc[np.isfinite(df_clean[col]), col].max()
    
    # 2. Clamp Inf to Max Finite
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], max_val)
    print(f" -> Clamped {col} to {max_val}")

# Handle NaNs (Simple fill with 0 for this exercise)
df_clean = df_clean.fillna(0)
```

### 4.3 Feature Engineering: Encoding Categorical Variables

Machines need numbers, not strings. How we convert them changes the model's understanding of "Distance."

**The Hat vs. Shoe Problem (Label Encoding)**
If we Label Encode: `Hat=0`, `Shirt=1`, `Pants=2`, `Shoe=3`.
The model thinks a **Shirt (1)** is "closer" to a **Hat (0)** than a **Shoe (3)** is. 
Does that make sense? No. They are just different categories.

**Strategies:**
* **One-Hot Encoding:** Creates a column for every category (`is_hat`, `is_shoe`). *Problem:* Explodes dimensionality (65,000 ports = 65,000 columns).
* **Frequency Encoding (Our Choice):** Replace the category with "How often does it appear?". Port 80 appears 40% of the time, so value = 0.40. Preserves importance without exploding dimensions.

```{code-cell} ipython3
# Calculate frequency of each destination port
port_freq = df_clean['Destination Port'].value_counts(normalize=True)

# Map it to a new column
df_clean['Dest_Port_Freq'] = df_clean['Destination Port'].map(port_freq)

# Drop the original raw number to prevent model confusion
df_model = df_clean.drop(['Destination Port'], axis=1)

df_model[['Dest_Port_Freq', 'Label']].head()
```

### 4.4 Splitting & Scaling (Avoiding Data Leakage)

**CRITICAL RULE:** Never scale your data *before* splitting. 
If you scale the whole dataset, the "Mean" and "Std Dev" of the Test set leak into the Training set.

**Correct Order:**
1. Split Data.
2. Fit Scaler on **Train**.
3. Transform **Train**.
4. Transform **Test**.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

X = df_model.drop('Label', axis=1)
y = df_model['Label']

# STRATIFIED SPLIT: Ensures class distribution (Attack/Benign) is consistent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# SCALE
scaler = StandardScaler()

# Fit on TRAIN, Apply to TRAIN
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# Apply to TEST (Do not fit!)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("Data Split & Scaled Successfully.")
```

## 5. Modeling & Evaluation

We will progress from simple to complex models to see how they handle the data nuances.

+++

### A. Naive Bayes & Correlation
Naive Bayes assumes all features are independent. Let's check if that assumption holds for network traffic.

```{code-cell} ipython3
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Train NB
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

print("Naive Bayes Results:")
print(classification_report(y_test, y_pred_nb))

# Visualization: Why might NB fail? Correlation.
# We only plot a subset of features to keep it readable.
corr_cols = ['Flow Duration', 'Total Fwd Packets', 'Flow Bytes/s', 'Packet Length Std']
corr_matrix = X_train[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation (Independence Assumption Check)")
plt.show()
```

### B. Decision Trees (Interpretability)
Trees allow us to see exactly *why* a decision was made. This is crucial for SOC analysts.

```{code-cell} ipython3
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Train Tree (Limit depth for readability)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train) # Note: Trees don't strictly require scaling, but we use raw for interpretability

plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X_train.columns, class_names=dt.classes_, filled=True, fontsize=12)
plt.title("Decision Tree Logic (Max Depth 3)")
plt.show()
```

### C. Random Forest & Feature Importance
Random Forests ensemble many trees to prevent overfitting. They also give us "Feature Importance" - telling us which logs matter most.

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier

# Train RF
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Extract Feature Importance
importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
})

# Sort and Top 10
top_10 = importances.sort_values(by='Importance', ascending=False).head(10)

# Plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Importance', y='Feature', data=top_10, palette='magma')
plt.title('Top 10 Features for PortScan Detection (Random Forest)')

# Add labels
for i in ax.containers:
    ax.bar_label(i, fmt='%.3f', padding=3)
plt.show()
```

### D. XGBoost (Gradient Boosting)
XGBoost is an evolution of Random Forests. 

**Key Differences:**
1.  **Sequential vs. Parallel:** Random Forests build trees independently (Parallel). XGBoost builds them one by one (Sequential).
2.  **Error Correction:** Each new tree in XGBoost tries to fix the **errors (residuals)** of the previous tree. It focuses on the "hard" cases.
3.  **Weak Learners:** It uses small, shallow trees (weak learners) that combine to form a strong predictor.

```{code-cell} ipython3
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# XGBoost requires numerical labels (0, 1)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Train XGB
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train_enc)

# Predict
y_pred_xgb = xgb_model.predict(X_test_scaled)

print("XGBoost Results:")
print(classification_report(y_test_enc, y_pred_xgb, target_names=le.classes_))
```

### E. Hyperparameter Tuning (Automated Optimization)

Simply training a model isn't enough. We must tune the "Hyperparameters" (settings) to get the best performance.

**Grid Search:** Tries *every combination* (Slow, exhaustive).
**Random Search:** Tries random combinations (Fast, usually good enough).

We will use `RandomizedSearchCV` with 3-Fold Cross Validation.

```{code-cell} ipython3
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid to search
param_dist = {
    'n_estimators': [50, 100, 200],         # Number of trees
    'max_depth': [3, 5, 10],                # Max depth of each tree
    'learning_rate': [0.01, 0.1, 0.3],      # Step size optimization
    'subsample': [0.7, 0.9, 1.0]            # Fraction of data to use per tree
}

print("Starting Randomized Search (This may take a minute)...\n")

# Initialize Search
random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    n_iter=5,              # Try 5 random combinations
    cv=3,                  # 3-Fold Cross Validation
    verbose=1,
    n_jobs=-1,             # Use all CPU cores
    random_state=42
)

# Fit
random_search.fit(X_train_scaled, y_train_enc)

print("\nBest Parameters found:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

## 6. Operational Realities: Deployment & Trust

Before we start the activity, a note on using these models in a real SOC:

### 1. Alert Fatigue
Even with 99% accuracy, if you have 1 million events per day, that is **10,000 false alarms**. 
Models must be tuned for **High Precision** to avoid burning out analysts.

### 2. Concept Drift
Cyber attacks evolve. A model trained on Port Scans from 2017 (like this one) will fail against modern techniques.
You must continuously retrain models (ML Ops).

### 3. Explainability
An analyst cannot block an IP just because the "Black Box" said so. We need Feature Importance (which we generated) to tell them *why* (e.g., "Blocked due to high destination port frequency").

+++

## 7. In-Class Activity: The Optimization Challenge

**Goal:** Improve upon the baseline XGBoost model provided above.

**Instructions:**
1.  **Feature Selection:** Use the Feature Importance chart above. Select only the **Top 5** features. Does accuracy drop? Does speed improve?
2.  **Hyperparameter Tuning:** Use the code block in **Section E** to find the optimal settings for your specific top-5 features. 
3.  **Data Split:** Try changing the split from `Stratified` to `Temporal` (simulate training on Monday's data and testing on Tuesday's).

**Submit:** Your best F1-Score and the list of features used to the class leaderboard.

+++

## Next Week: Unsupervised Learning
What happens when we don't have labels? What if we don't know it's a "PortScan"?

Next week we cover **Clustering (K-Means, DBSCAN)** and Anomaly Detection.
