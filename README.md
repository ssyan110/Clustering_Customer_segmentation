# Customer Segmentation with K-Means Clustering (PySpark)

This project performs customer segmentation on a credit card dataset using K-Means clustering in PySpark. By grouping users based on their financial behavior, the goal is to provide insights that can support targeted marketing, personalized services, and strategic decision-making in financial institutions.

## Objectives

- Preprocess real-world customer transaction data
- Apply clustering techniques to segment users into distinct groups
- Use dimensionality reduction and scaling to improve model quality
- Visualize and evaluate the effectiveness of clustering results

## Technologies Used

- PySpark (MLlib, DataFrame APIs)
- K-Means Clustering
- PCA (Principal Component Analysis)
- RobustScaler
- Seaborn, matplotlib, pandas
- Jupyter Notebook

## Workflow Overview

### 1. Data Preparation
- Loaded credit card data from `CC_GENERAL.csv`
- Dropped non-numerical IDs
- Handled missing values by imputing median values

### 2. Feature Selection & Scaling
- Selected top 4 features based on variance
- Scaled features using `RobustScaler` to minimize outlier influence

### 3. Dimensionality Reduction
- Applied PCA to reduce the feature space to 2 dimensions for better visualization and clustering performance

### 4. Clustering
- Used Elbow Method to determine optimal number of clusters (k=3)
- Trained K-Means model using `pca_features`
- Assigned cluster labels to each customer

### 5. Evaluation
- Visualized clusters in 2D using Seaborn
- Evaluated model using **Silhouette Score**: `0.89`, indicating strong cluster separation and cohesion

## Results

- Optimal number of clusters: **3**
- Key behavioral groupings were identified, supporting segmentation-based decision-making
- Visual plots clearly demonstrated cluster separation

## Files Included

- `Customer_Clustering.ipynb` – Full code, step-by-step process
- Dataset: `CC_GENERAL.csv`

## How to Run

1. Ensure you have PySpark and required Python libraries installed
2. Update the path to your dataset if needed
3. Open the notebook in Jupyter
4. Run each cell sequentially

## Example Use Cases

- Financial institutions segmenting customers for credit offers
- Retail businesses personalizing campaigns based on spending patterns
- Analysts exploring unsupervised learning on behavioral data
