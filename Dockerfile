FROM python:3.9.7-slim

RUN pip install pipenv

WORKDIR /app 




COPY ["Pipfile","Pipfile.lock","./"] 

RUN pipenv install --system --deploy

COPY ["model_rf.bin","predict.py","./"] 

