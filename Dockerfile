FROM ubuntu:18.04

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  python3-dev python3-pip -y
RUN apt-get update && apt-get install libopenblas-dev -y
COPY . /demo/
WORKDIR /demo
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
CMD python3 app.py