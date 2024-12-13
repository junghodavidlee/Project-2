# Credit Card Fraud Binary Classification

## Table of Contents
- [1. Introduction](#introduction)
- [2. Data Preparation](#data-preparation)
- [3. Data Cleaning and Transformation](#data-cleaning-and-transformation)
- [4. Exploratory Data Analysis](#exploratory-data-analysis)
- [5. Data Preprocessing](#data-preprocessing)
- [6. Define Machine Learning Models](#define-machine-learning-models)
- [7. Model Analysis](#model-analysis)
- [8. Conclusion](#conclusion)

## 1. Introduction <a id="introduction"></a>

This notebook analyzes banking data with the aim of understanding the markers of fraudulent credit card transactions by creating multiple machine learning models and finding the most accurate and precise models.
- [Presentation](https://docs.google.com/presentation/d/1srQSV0zAgFcFW_pgsdZ4xFeoHwl57mfcbHWBF52qLS8/edit?usp=sharing)

### Data Sources
- [Kaggle Credit Card Transaction Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)

### 2. Data Preparation <a id="data-preparation"></a>
- **Objective**: Import necessary libraries and load dataset for analysis.
- **Actions**: 
  - Loaded and prepare credit card transaction dataset.

### 3. Data Cleaning and Transformation <a id="data-cleaning-and-transformation"></a>
- **Objective**: Prepare the data for analysis by cleaning and transforming it.
- **Actions**:
  - Dropped unnecessary columns.
  - Filled missing merchant zip code data with customer zip code data.
  - Scaled numerical data using Standard Scaler
  - Encoded categorical data using a variety of techniques, applicably chosen between One Hot Encoding, label Encoding, and frequency encoding to best fit each of the instances of data.

  ### 4. Exploratory Data Analysis <a id="exploratory-data-analysis"></a>
- **Objective**: Perform exploratory analysis to uncover patterns and insights.
- **Actions**:
  - Isolate fraudulent data from rest of the dataset to perform analysis
  - Created bar charts, pie charts and heatmaps to identify patterns within the fraudulent data
  - Strongest correlative data represented in Date/Time data patterns and geographic location
  

### 5. Data Preprocessing <a id="data-preprocessing"></a>
- **Objective**: Find the most suitable methods of preprocessing our highly imbalanced, large dataset for modeling
- **Actions**:
  - Tested between various preprocessing methods
  - Decided that SMOTE preprocessing was most applicable for an imbalanced dataset
  - In the process of encoding the data in part 3, many columns were added so PCA was conducted to reduce the dimensionality of the data.


### 6. Define Machine Learning Models <a id="define-machine-learning-models"></a>
- **Objective**: Identify machine learning models that are most effective in binary classification of credit card fraud
- **Actions**:
  - Defined and analyzed 7 different models, each with unprocessed, resampled, and pca data
  - Narrowed down effective models to Stochastic Gradient Descent Classification, Decision Tree Classification, Extreme Gradient Boosting, and Random Forest Classification.

### 7. Model Analysis <a id="model-analysis"></a>
- **Objective**: Analyze the usefulness of each of the models.
- **Actions**:
  - Evaluate the accuracy score, confusion matrix and classification report for each of the models, and identify the ones with the highest scores for each of the various metrics.

### 8. Conclusion <a id="conclusion"></a>
- After exploring the data, it was clear that patterns existed within the most common timeframes for fraudulent transactions. Namely the hours between 10pm and 4am, particularly on weekends, were when it was most likely for these instances to occur. We were also able to identify the most common categories in which credit card transaction fraud occurred. After creating several models to analyze our data, we noticed nearly every single model had an accuracy score of 99%. This was because our data has 1.3 million rows, of which only roughly 7 thousand are fraudulent transactions. Because of the highly unbalanced nature of our data, we were focused on the precision and recall scores for fraudulent cases. Of the various models tested, we concluded that Decision Tree Classifier, Extreme Gradient Boosting, and Random Tree Classifier were the best modeling methods, giving us F1 scores upwards of 70%.
