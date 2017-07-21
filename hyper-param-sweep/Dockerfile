FROM tensorflow/tensorflow:latest-gpu-py3
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN mkdir ./output
RUN mkdir ./logs
RUN mkdir ./checkpoints
RUN pip install -r requirements.txt
COPY ./* /app/