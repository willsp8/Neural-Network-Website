FROM python 

EXPOSE 5000

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

ENV FLASK_APP=main.py

EXPOSE 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]



