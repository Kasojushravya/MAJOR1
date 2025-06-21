from flask import Flask, render_template, request
from utils.predict_symptoms import predict_by_symptoms
from utils.predict_testvalues import predict_by_test_values

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptom')
        result = predict_by_symptoms(symptoms)
        return render_template('result.html', all_diseases=result)
    return render_template('symptom_form.html')

@app.route('/test-values', methods=['GET', 'POST'])
def test_values():
    if request.method == 'POST':
        values = dict(request.form)
        result = predict_by_test_values(values)
        return render_template('result.html', all_diseases=result)
    return render_template('testvalue_form.html')

@app.route('/assistance')
def assistance():
    return render_template('assistance.html')

if __name__ == '__main__':
    app.run(debug=True)
