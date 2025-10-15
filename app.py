from flask import Flask,render_template,request
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            iq =  request.form.get('iq'),
            prev_sem_result =  request.form.get('prev_sem_result'),
            cgpa =  request.form.get('cgpa'),
            academic_performance =  request.form.get('academic_performance'),
            extra_curricular_score =  request.form.get('extra_curricular_score'),
            communication_skills =  request.form.get('communication_skills'),
            projects_completed =  request.form.get('projects_completed'),
            internship_experience = request.form.get('internship_experience')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        ans = 'No'
        if results[0] == 1:
            ans = 'Yes'
        
        return render_template('home.html',results=ans)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)