#change from for previous image.
FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt
COPY api /api
COPY data_scientist_skills /data_scientist_skills
COPY desc_class_model.joblib /desc_class_model.joblib
COPY jobs_rec_model.joblib /jobs_rec_model.joblib

RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
