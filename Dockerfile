FROM python:3.9.7-slim

RUN pip install pipenv

WORKDIR /app 




COPY ["Pipfile","Pipfile.lock","./"] 

RUN pipenv install --system --deploy

COPY ["model_C=10.bin","predict.py","./"] 

