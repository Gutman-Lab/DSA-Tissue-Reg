# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim-buster

# Copy local code to the container image.
ENV APP_HOME /app
ENV PYTHONUNBUFFERED True
WORKDIR $APP_HOME

# Install Python dependencies and Gunicorn
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir "uvicorn[standard]" gunicorn
RUN groupadd -r app && useradd -r -g app app

RUN pip install girder-client pymongo
RUN mkdir /home/app
RUN chmod -R 777 /home/app

# Copy the rest of the codebase into the image
# COPY --chown=app:app . ./  ## Don't do this now during debug..

## It's possible if I don't build on a MAC I may need not this.. TODO is change to linux/amd64 build
RUN apt-get update
RUN apt install libgl1-mesa-glx -y
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
USER app

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available in Cloud Run.
# CMD exec gunicorn --bind :6667 --log-level info --workers 1 --worker-class uvicorn.workers.UvicornWorker --threads 3 --timeout 0 --reload app:server
EXPOSE 8050
#CMD exec gunicorn --bind 0.0.0.0:6667 --log-level info --workers 1 --timeout 0 --reload app:server
CMD ["gunicorn","-b","0.0.0.0:8050","--reload","app:server"]
# gunicorn --bind 0.0.0.0:6667 --log-level info --workers 1 --timeout 0 --reload app:server
### Had put port in there 