FROM python:3.12.2

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD ["python3"]