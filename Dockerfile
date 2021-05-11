FROM ubuntu:focal

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
RUN pip3 install numpy==1.19
RUN pip3 install jupyter
RUN pip install jupyterlab
RUN pip3 install /usr/local/dist/CytoPy-2.0.1-py3-none-any.whl

### Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

