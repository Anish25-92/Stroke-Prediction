from flask import Flask, render_template, request
import joblib
import os
import  numpy as np
import pickle

app= Flask(__name__)
model=joblib.load('models/rf.sav')
scaler_file= open('models/scaler.pkl','rb')
scaler=pickle.load(scaler_file)



@app.route("/")
def index():
    return render_template("index.html")

@app.route('/prediction', methods=['POST'])
def prediction():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heartdisease = int(request.form['heartdisease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    answer = predictAnswer(model, gender, age, hypertension, heartdisease, ever_married, work_type, Residence_type, avg_glucose_level,bmi,smoking_status)
    return render_template('prediction.html', answer=answer)

def predictAnswer(model, gender, age, hypertension, heartdisease, ever_married, work_type, Residence_type, avg_glucose_level,bmi,smoking_status): 
    row = [gender, age, hypertension, heartdisease, ever_married, work_type, Residence_type, avg_glucose_level,bmi,smoking_status]
    data=[np.array(row)]
    data=scaler.transform(data)
    result = model.predict(data)
    result = result[0]
    if result == 1:
        return "Yes, you may be likely to have a stroke. Please see a doctor."
    elif result == 0:
        return "No, you are not likely to have a stroke" 
    else:
        return "A problem has occured in processing your data. Please reach out to Alex. She promises it was working before :("
    


if __name__=="__main__":
    app.run(debug=True,port=7384)
