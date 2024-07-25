# StreamBoard
Streamlit Dashboard of Machine Learning models for Obesity Classification Dataset. The app has been deployed at https://blackmoonbear-streamboard.streamlit.app/

## Overview

Streamboard is an interactive Streamlit dashboard designed to visualize and predict obesity probability based on lifestyle and health data. The dashboard provides features for data overview, model building and evaluation, and making predictions.

## Features

- **Data Overview**: Visualize the dataset attributes, distributions, pair plots, and correlation heatmaps.
- **Model Building and Evaluation**: Train and evaluate different classification models (RandomForestClassifier, XGBClassifier, LGBMClassifier) with various hyperparameters.
- **Make Predictions**: Use trained models to predict obesity probability for new data inputs and interpret the results with LIME and SHAP explanations.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/VIROOPAKSHC/streamboard.git
    cd streamboard
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

## Usage

1. **Data Overview**:
    - View and explore the dataset.
    - Visualize distributions, pair plots, and correlation heatmaps.
    - Inspect column descriptions and statistics.

2. **Model Building and Evaluation**:
    - Select a model type (RandomForestClassifier, XGBClassifier, LGBMClassifier).
    - Adjust hyperparameters and train the model.
    - Evaluate model performance with metrics, classification reports, confusion matrices, and feature importance charts.

3. **Make Predictions**:
    - Fill in the inputs to make predictions.
    - View model predictions and confidence levels.
    - Interpret the predictions using LIME and SHAP explanations.

## Dataset

The dataset used in this project is the Obesity Dataset. You can find the dataset [here](https://www.kaggle.com/datasets/ikjotsingh221/obesity-risk-prediction-cleaned/data).

## Dependencies

The project relies on the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lime
- shap
- xgboost
- lightgbm
- plotly
- pdpbox
- streamlit

## Acknowledgements

This project was made with ❤️ by [Chekuri Viroopaksh](https://www.linkedin.com/viroopaksh-chekuri). Connect with me on [LinkedIn](https://www.linkedin.com/viroopaksh-chekuri) and check out my other projects on [GitHub](https://github.com/VIROOPAKSHC).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
