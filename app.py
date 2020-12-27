import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd 


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = [x for x in request.form.values()]
    #final_features = [np.array(features)]
    test_data=pd.DataFrame({'Gender':final_features[0], 'Married':final_features[1], 'Dependents':final_features[2], 'Education':final_features[3],'Self_Employed':final_features[4], 'ApplicantIncome':final_features[5], 'CoapplicantIncome':final_features[6], 'LoanAmount':final_features[7],'Loan_Amount_Term':final_features[8], 'Credit_History':final_features[9], 'Property_Area':final_features[10]},index=[0])
    test_data['Gender']=test_data['Gender'].apply(lambda x:1 if x=='Male' else 0)


    test_data['Married']=test_data['Married'].apply(lambda x:1 if x=='Yes' else 0)


    test_data['Education']=test_data['Education'].apply(lambda x:1 if x=='Graduate' else 0)


    test_data['Self_Employed']=test_data['Self_Employed'].apply(lambda x:1 if x=='Yes' else 0)


    test_data['Property_Area']=test_data['Property_Area'].apply(lambda x:1 if x=='Rural' else (2 if x=='Semiurban' else 3))


    test_data['Dependents']=test_data['Dependents'].apply(lambda x:1 if x=='1' else (2 if x=='2' else (3 if x=="3+" else 0)))
    
    prediction = model.predict(test_data)
    if prediction[0]==1:
        str="Approved"
    else:
        str=" Not Approved"

    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Loan  {}'.format(str))


if __name__ == "__main__":
    app.run(debug=True)