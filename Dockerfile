FROM mongo:bionic

### Update and install Python
# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev
RUN apt-get install -y curl python3.8 python3.8-dev python3.8-distutils

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.8

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

### Install CytoPy & Jupyter Notebook
WORKDIR /usr/local/
COPY dist/ dist/
RUN pip3 install wheel
RUN pip3 install jupyter
RUN pip3 install numpy==1.19
RUN pip3 install /usr/local/dist/CytoPy-2.0-py3-none-any.whl

### Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]





