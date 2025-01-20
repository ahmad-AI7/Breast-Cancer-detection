# Breast-Cancer-detection
Machine learning project to predict the diagnosis of breast cancer
# Breast Cancer Prediction

This project aims to predict whether a breast cancer tumor is benign (not cancerous) or malignant (cancerous) using machine learning.

## Dataset

The dataset used for this project is the Breast Cancer Wisconsin (Diagnostic) Data Set, which can be found on UCI Machine Learning Repository. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.

## Methodology

1. **Data Loading and Preprocessing:** The dataset is loaded using Pandas. Missing values are handled, and categorical features are encoded using Label Encoding.
2. **Data Splitting:** The dataset is split into training and testing sets using `train_test_split` from scikit-learn.
3. **Feature Scaling:** Features are scaled using StandardScaler to ensure that all features have a similar range of values.
4. **Model Training:** A Logistic Regression model is trained on the training data.
5. **Model Evaluation:** The model's performance is evaluated using accuracy score.
6. **Prediction System:** A prediction system is built to predict the diagnosis for new input data.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Pickle

## Usage

1. Install the required dependencies.
2. Load the dataset.
3. Preprocess the data.
4. Train the model.
5. Evaluate the model.
6. Use the prediction system to predict the diagnosis for new input data.

## Results

The Logistic Regression model achieved an accuracy of approximately 97% on the testing data.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License.
