FROM python:slim

RUN mkdir /src

WORKDIR /src

COPY setup_funcs.py ./

CMD python setup_funcs.py