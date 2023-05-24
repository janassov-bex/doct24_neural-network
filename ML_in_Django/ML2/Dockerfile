FROM python:3.10-bullseye
WORKDIR /app
RUN apt-get update
RUN pip install --upgrade pip
COPY . /app/.
RUN pip install -r requirements.txt
WORKDIR /app/MLMedic
CMD ["gunicorn", "MLMedic.wsgi:application", "--bind", "0.0.0.0:3000"]
EXPOSE 3000