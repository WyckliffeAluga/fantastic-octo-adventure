
FROM continuumio/anaconda3:4.4.0                                                                                          """
MAINTAINER UNP, https://unp.education
COPY ./app /usr/local/python/                                                                                                      """
EXPOSE 5000                                                                                                                                       import pickle
WORKDIR /usr/local/python/                                                                                                                                from flasgger import Swagger
RUN pip install -r requirements.txt                                                                                                                               import pandas as pd
CMD python iris_api.py  
