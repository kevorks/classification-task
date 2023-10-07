FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/src

CMD [ "python", "main.py" ]