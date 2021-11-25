FROM python:3.7-slim

RUN mkdir /src

WORKDIR /src

COPY setup_funcs.py ./

RUN pip install xarray==0.14.0 numpy==1.16.5 pandas==1.0.1 netcdf4==1.5.1.2

CMD python setup_funcs.py