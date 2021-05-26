FROM python:3.8-slim-buster

EXPOSE $FASTAPI_PORT

WORKDIR /app

RUN apt update && apt install -y ffmpeg libsm6 libxext6

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "./src/main.py"]