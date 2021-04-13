FROM ubuntu:focal

# Update and install python
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.8 python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN apt-get update -y

### Install CytoPy & Jupyter Notebook
WORKDIR /usr/local/
COPY dist/ dist/
RUN pip3 install wheel
RUN pip3 install /usr/local/dist/CytoPy-2.0-py3-none-any.whl
RUN






