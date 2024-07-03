from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
lr_loaded = joblib.load('logistic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])
    type_ = request.form['type']

    # One-hot encode the transaction type
    type_CASH_IN = 1 if type_ == "CASH_IN" else 0
    type_CASH_OUT = 1 if type_ == "CASH_OUT" else 0
    type_DEBIT = 1 if type_ == "DEBIT" else 0
    type_PAYMENT = 1 if type_ == "PAYMENT" else 0
    type_TRANSFER = 1 if type_ == "TRANSFER" else 0

    input_data = pd.DataFrame({
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'isFlaggedFraud': [0],  # Assuming 0 for simplicity
        'type_CASH_IN': [type_CASH_IN],
        'type_CASH_OUT': [type_CASH_OUT],
        'type_DEBIT': [type_DEBIT],
        'type_PAYMENT': [type_PAYMENT],
        'type_TRANSFER': [type_TRANSFER]
    })

    prediction = lr_loaded.predict(input_data)
    output = "fraudulent" if prediction[0] == 1 else "not fraudulent"

    return render_template('index.html', prediction_text='The transaction is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

