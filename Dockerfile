FROM python:3.12-slim


WORKDIR /app

COPY requirements.txt /app/


RUN pip install -r requirements.txt


COPY HomePage.py /app/
COPY pages /app/pages/
COPY projectData /app/projectData/
COPY models /app/models/
COPY data/processed/viz_df.csv /app/data/processed/
COPY data/raw/appartments.csv /app/data/raw/

EXPOSE 5000


CMD ["streamlit","run","HomePage.py"]