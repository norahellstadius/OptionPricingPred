FROM python:3.11
WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install tensorflow
RUN pip install --no-cache-dir -r requirements.txt 

ENTRYPOINT ["/bin/bash"]


