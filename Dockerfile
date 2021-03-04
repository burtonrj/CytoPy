FROM ubuntu:latest

# Update and install Python
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN pip3 -q install pip --upgrade

# Install MongoDB
RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add - && \
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" \
    | tee /etc/apt/sources.list.d/mongodb-org-4.4.list
RUN apt-get update && \
    apt-get install -y mongodb-org && \
    apt-get update

# Install CytoPy & Jupyter Notebook
RUN pip3 install numpy && \
    pip3 install cytopy && \
    pip3 install jupyter

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# When container runs, start mongodb service and launch Jupyter Notebooks
CMD ["systemctl start mongod"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]



