FROM jupyter/scipy-notebook
### Install CytoPy & Jupyter Notebook
WORKDIR /usr/local/
COPY dist/ dist/
RUN pip3 install wheel
RUN pip3 install /usr/local/dist/CytoPy-2.0-py3-none-any.whl






