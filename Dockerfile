FROM ubuntu:focal
ARG GIT_COMMIT
ARG BUILD_DATE

### Update and install Python
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

### Install CytoPy & Jupyter Notebook
WORKDIR /usr/local/
COPY dist/ dist/
RUN pip3 install wheel
RUN pip3 install jupyter
RUN pip install jupyterlab
RUN pip3 install /usr/local/dist/CytoPy-2.1.0-py3-none-any.whl

### Labels
LABEL version="2.1.0"
LABEL commit=$GIT_COMMIT
LABEL build_date=$BUILD_DATE

### Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

