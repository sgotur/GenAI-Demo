FROM python:3.9-slim

WORKDIR /usr/src/app

RUN rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get upgrade -y
RUN apt-get install graphviz git -y

COPY static/*.whl .
COPY *.txt .

RUN pip3 install boto3-1.26.142-py3-none-any.whl
RUN pip3 install botocore-1.29.142-py3-none-any.whl
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm -f ./*.whl

COPY *.txt .
COPY *.py .
COPY *.jpg .
COPY static/ static/
COPY utils/ utils/
COPY pages/ pages/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT [ "streamlit", "run", "Welcome.py", \
             "--logger.level", "info", \
             "--browser.gatherUsageStats", "false", \
             "--browser.serverAddress", "0.0.0.0", \
             "--server.enableCORS", "false", \
             "--server.enableXsrfProtection", "false", \
             "--server.port", "8501"]