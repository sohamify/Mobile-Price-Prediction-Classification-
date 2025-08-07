# Mobile Price Range Prediction: A Bagging and Random Forest Approach

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Dataset](#2-dataset)
3.  [Methodology](#3-methodology)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Building](#model-building)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
4.  [Results and Conclusion](#4-results-and-conclusion)
5.  [How to Run the Code](#5-how-to-run-the-code)
6.  [Dependencies](#6-dependencies)

---

### 1. Project Overview

This project aims to build and compare two powerful ensemble machine learning models, **Bagging Classifier** and **Random Forest Classifier**, to predict the price range of a mobile phone based on its technical specifications.

The process follows a standard machine learning workflow:
-   **In-depth Exploratory Data Analysis (EDA)** to understand the data's structure and relationships.
-   **Data preprocessing** to prepare features for modeling.
-   **Model building** and evaluation of two distinct ensemble models.
-   **Hyperparameter tuning** using `GridSearchCV` to optimize each model's performance.
-   **Final model selection** based on a comprehensive performance comparison.

---

### 2. Dataset

The project uses the `train.csv` dataset, which contains 20 features describing a phone's specifications and one target variable, `price_range`.

**Features:**
-   `battery_power`: Total energy a battery can store in mAh.
-   `blue`: Has Bluetooth or not.
-   `clock_speed`: Speed at which microprocessor executes instructions.
-   `dual_sim`: Has dual SIM support or not.
-   `fc`: Front Camera megapixels.
-   `four_g`: Has 4G or not.
-   `int_memory`: Internal Memory in GB.
-   `m_dep`: Mobile depth in cm.
-   `mobile_wt`: Mobile weight.
-   `n_cores`: Number of cores of the processor.
-   `pc`: Primary Camera megapixels.
-   `px_height`: Pixel resolution height.
-   `px_width`: Pixel resolution width.
-   `ram`: Random Access Memory in MB.
-   `sc_h`: Screen height in cm.
-   `sc_w`: Screen width in cm.
-   `talk_time`: Longest time a single battery charge will last when talking.
-   `three_g`: Has 3G or not.
-   `touch_screen`: Has a touch screen or not.
-   `wifi`: Has WiFi or not.

**Target Variable:**
-   `price_range`: The price range of the mobile phone. This is a categorical variable with four classes:
    -   `0` (low cost)
    -   `1` (medium cost)
    -   `2` (high cost)
    -   `3` (very high cost)

**Key Insights from EDA:**
-   The dataset contains **no missing values**, which simplifies the preprocessing pipeline.
-   The **target classes (`price_range`) are perfectly balanced**, with 500 instances for each class. This means no class imbalance techniques are required.
-   **`ram` is identified as the most crucial feature**, showing a very strong positive correlation with the price range.
-   Other features like **`battery_power`**, **`px_height`**, and **`px_width`** also show clear positive trends with increasing price range.
-   Features such as `pc` (Primary Camera) and `fc` (Front Camera) appear to have a weaker correlation with the final price.

---

### 3. Methodology

#### Exploratory Data Analysis (EDA)
-   **Univariate Analysis:** Histograms were used to visualize the distribution of each feature, checking for skewness or potential outliers.
-   **Bivariate Analysis:** Boxplots were generated to understand how the distribution of each numerical feature varies across the four `price_range` classes.
-   **Correlation Matrix:** A heatmap was used to visualize the correlation between all features, highlighting important relationships and potential multicollinearity.

#### Data Preprocessing
-   The data was split into a **70% training set and a 30% testing set** using `train_test_split`.
-   The `stratify=y` parameter was used to ensure that the proportion of each price class was maintained in both the training and testing sets.
-   A `StandardScaler` was applied to all numerical features to standardize their ranges.

#### Model Building
Two ensemble models were built and evaluated:

1.  **Bagging Classifier:** An ensemble of **Decision Trees** trained on bootstrap samples of the data. This method helps to reduce the model's variance and prevent overfitting.
2.  **Random Forest Classifier:** An extension of Bagging that adds an extra layer of randomness. For each split, it considers only a **random subset of features**, which further decorrelates the trees and improves performance.

#### Hyperparameter Tuning
-   `GridSearchCV` with **5-fold cross-validation** was used to find the optimal hyperparameters for both the Bagging and Random Forest models.
-   Parameters tuned for **Bagging Classifier** included `n_estimators`, `max_samples`, and the `max_depth` of its base Decision Tree estimator.
-   Parameters tuned for **Random Forest Classifier** included `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`.

---

### 4. Results and Conclusion

The performance of the initial and tuned models was evaluated using **accuracy**, a **classification report**, and a **confusion matrix**. The results clearly demonstrate the effectiveness of hyperparameter tuning in improving model performance.

Based on the final accuracy and the balance of precision and recall across all price ranges, the **tuned Random Forest Classifier is expected to be the superior model** for this specific problem. Its ability to decorrelate individual trees through feature randomization typically gives it a slight edge over a standard Bagging Classifier.

---

### 5. How to Run the Code
1.  Ensure you have the `train.csv` file in the same directory as your Python script.
2.  Install all the required dependencies listed below.
3.  Execute the Python script in your environment (e.g., Jupyter Notebook, VS Code, or from the terminal).

---

### 6. Dependencies
This project requires the following Python libraries. You can install them using pip:
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
