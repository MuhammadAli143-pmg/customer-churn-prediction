# Customer Churn Predictor
Project App Url: https://customer-churn-predictiongit-jaynzkr8nhugcwgqxxh4fq.streamlit.app/
A portfolio project that explores telecom customer churn, trains a recall-focused machine-learning pipeline, and provides a Streamlit interface for individual customer predictions.

## Project files

- `Customer_Churn_Predictor.ipynb` — Google Colab/Jupyter notebook with each project section in its own cell.
- `app.py` — Streamlit application.
- `churn_model.joblib` — trained preprocessing + classification pipeline (created by running the notebook).
- `Telco-Customer-Churn.csv` — supplied dataset.
- `requirements.txt` — Python dependencies.

## Run the notebook

1. Open `Customer_Churn_Predictor.ipynb` in Google Colab.
2. Upload `Telco-Customer-Churn.csv` when prompted, or place it in the same folder.
3. Run all cells from top to bottom.
4. Download `churn_model.joblib` from the Colab file panel after the final save cell.

## Run the Streamlit app locally

Place `churn_model.joblib` beside `app.py`, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model workflow

1. Load the Telco Customer Churn dataset.
2. Inspect shape, data types, missing values and duplicates.
3. Convert `TotalCharges` to numeric, fill its 11 blank values with zero, and drop `customerID`.
4. Explore churn by contract, tenure, monthly charges and payment method.
5. Split into 80% training and 20% testing using stratification.
6. Scale numerical features and one-hot encode categorical features in a pipeline.
7. Compare Logistic Regression, Decision Tree and Random Forest.
8. Use balanced class weights and compare recall.
9. Tune the highest-recall initial model with GridSearchCV using recall scoring.
10. Evaluate on the held-out test set with a confusion matrix and classification report.
11. Derive business actions from observed churn patterns.

## Dataset

Telco Customer Churn dataset supplied for the PSEB AI Bootcamp project. The CSV contains 7,043 customer records and 21 columns.

## Important notes

- The risk bands in the app (Low, Medium, High) are illustrative probability thresholds, not calibrated business decisions.
- Model performance depends on the train/test split and the tuning process. Use the notebook's measured results rather than assuming a fixed accuracy or recall.
- Associations in exploratory analysis do not establish causation.
- Before deployment, validate the model, thresholds, privacy practices and retention actions with the business.

## Deployment

1. Create a public GitHub repository.
2. Upload the notebook, `app.py`, `requirements.txt`, dataset, `churn_model.joblib`, and this README.
3. In Streamlit Community Cloud, create an app from the repository and select `app.py`.
4. Add the deployed app URL near the top of this README.
5. Submit the live app link, GitHub repository link and LinkedIn post link to the LMS.
