#!/bin/bash
## This will rebuild the container, and start the app, and copy/bind the current working directory into /app for quick reload
docker stop tissue-reg
docker rm tissue-reg
docker build -t tissue-reg .
docker run -it -p 8050:8050 -v $(pwd):/app --name tissue-reg tissue-reg /bin/bash
