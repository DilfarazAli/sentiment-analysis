# Sentiment Analysis of Tweets

## Description
This project demonstrates a sentiment analysis model trained on a large dataset of tweets to classify them as positive or negative.

## Dataset
The model is trained on the Sentiment140 dataset, which contains 1.6 million tweets. You can find more information about the dataset [here](https://www.kaggle.com/datasets/kazanova/sentiment140).

## Methodology
1.  **Data Loading and Exploration:** Loading the dataset and checking for missing values.
2.  **Data Preprocessing:** Cleaning and preprocessing the text data, including removing non-alphabetic characters, converting to lowercase, splitting words, removing stopwords, and applying stemming. The target variable is also converted to a binary representation (0 for negative, 1 for positive).
3.  **Data Splitting:** Dividing the dataset into training and testing sets.
4.  **Feature Extraction:** Converting text data into numerical feature vectors using TF-IDF.
5.  **Model Training:** Training a Logistic Regression model on the training data.
6.  **Model Evaluation:** Assessing the model's performance using accuracy score on both training and testing data.
7.  **Model Saving and Loading:** Saving the trained model using `pickle` and loading it back.
8.  **Prediction:** Making predictions on new data using the loaded model.

## Requirements
*   `pandas`
*   `numpy`
*   `re`
*   `nltk`
*   `sklearn`
*   `kaggle`

## How to Run the Code
1.  Open the `sentiment_analysis_notebook.ipynb` file in Google Colab or any other Jupyter-compatible environment.
2.  Install the necessary libraries by running the first code cell (`! pip install kaggle` is already included, you might need to install others if not already present).
3.  Download the dataset:
    *   Obtain a Kaggle API key by going to your Kaggle account settings.
    *   Upload your `kaggle.json` file when prompted after running the cell that uses `files.upload()`.
    *   The notebook includes cells to set up the Kaggle directory and download the dataset.
4.  Run the remaining code cells in sequence to preprocess the data, train the model, and make predictions.

## Results
The trained model achieved an accuracy of approximately [Your Test Accuracy Score]% on the test data. (You can replace this with the actual accuracy score from your notebook output).

## Files
*   `sentiment_analysis_notebook.ipynb` (This notebook file)

## Author
[Dilfaraz Ali]
