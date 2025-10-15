import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)


class CustomData():
    def __init__(self,
                 iq:int, 
                 prev_sem_result:int, 
                 cgpa:int, 
                 academic_performance:int, 
                 extra_curricular_score:int, 
                 communication_skills:int, 
                 projects_completed:int, 
                 internship_experience:str):
        
        self.iq =  iq
        self.prev_sem_result =  prev_sem_result
        self.cgpa =  cgpa
        self.academic_performance =  academic_performance
        self.extra_curricular_score =  extra_curricular_score
        self.communication_skills =  communication_skills
        self.projects_completed =  projects_completed
        self.internship_experience = internship_experience

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'iq':[self.iq], 
                'prev_sem_result':[self.prev_sem_result], 
                'cgpa':[self.cgpa], 
                'academic_performance':[self.academic_performance], 
                'extra_curricular_score':[self.extra_curricular_score], 
                'communication_skills':[self.communication_skills], 
                'projects_completed':[self.projects_completed], 
                'internship_experience':[self.internship_experience]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)