# strokedetect
Capstone Project  part of ML Data Talks Club
Data has been downloaded from kaggle  https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/discussion/392190
This data used to predict heart stroke(yes/no) based having below categorcial and numerical Features.

      1.numerical=[ 'age',  'hypertension',  'heart_disease',  'avg_glucose_level',  'bmi','id']
      2.categorical=['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
      

We also have  "stroke" which is output variable and  predict using various modesl and below are :
Expermimented using below models:

       1.Logistic Regression Model but did not good ROC_AUC Score.
       2.Decison Tree
       3.Random Forest


> **Outcome of Models**:I trained the dataset using different models, tuned them with various hyperparameters, and found that Random Forest achieved a great ROC_AUC score after tuning.

Project Directory: All files are organized in the 'midterm_projects' folder:

      1.  'healthcare-dataset-stroke-data.csv': This is the dataset used for evaluating the models and making predictions. It's stored in the Git repository.
      2.notebook.ipynb': This file contains the complete code for all the models.
      3. 'model_C=10.bin': This folder contains the binary file for the Logistic Regression model, and it will be available in Docker.
      4. 'Pip Files': These files are used by Docker to install Python-dependent packages.
      5. predict.py:Used for opening pickled model file and predict proba of incoming patient record to predict stroke.
      5. model_rf: These model file is pickled for hypertuning Random Forest and used predict.py
      6.Dockerfile:Contains entire config to run docker container.
      7.train.py: Used for testing for one  recod without need of docker.
      8.Test_record_1031:Hold single patient info using container service 

How to test and predict record using docker container:
      1.Create midterm_projects Folder.
      2.Download docker file and all other files to above directory.
      3.docker build -t midtermproj .
      4.docker run -it --rm --entrypoint=bash midtermproj:latest
      5.gunicorn --bind 0.0.0.0:9696 predict:app
      6.Open Test_record_1031  and add json record  to test out.
      
      
      

  

