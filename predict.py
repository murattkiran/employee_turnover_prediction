import pickle

from flask import Flask, request, jsonify
import xgboost as xgb


model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('turnover')

@app.route('/predict', methods=['POST'])
def predict():
    employee = request.get_json()

    X = dv.transform([employee])
    dx = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
    prediction = model.predict(dx)[0]
    turnover = prediction >= 0.5

    result = {
        'turnover_probability': float(prediction),
        'turnover': bool(turnover)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)