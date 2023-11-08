import pickle
from flask import Flask
from flask import request
from flask import jsonify
C=10
model_file=f'model_rf.bin'



with open (model_file,'rb') as f_in:
    dv,model=pickle.load(f_in)


  
app=Flask("stroke")
  
@app.route('/predict',methods=['POST'])
def predict():

    patient=request.get_json()
    
    X=dv.transform(patient)
    stroke_predict=model.predict_proba(X)[:,1]
    result={
        'stroke_predicttion':float(stroke_predict)
    }
    return jsonify(result)


if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)

  



