FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]