'''
MeDIT.Log
Functions for Log.

author: Yang Song, Chengxiu Zhang.
All right reserved
'''

from glob import glob
import pandas as pd
import os
import logging
import time
from logging.handlers import RotatingFileHandler

class CustomerCheck:
    def __init__(self, store_path, patient=10, data={'State': [], 'Path': []},
                 rewrite=False, follow_write=False):
        if isinstance(data, list):
            self.data = pd.DataFrame(columns=data)
        elif isinstance(data, dict):
            self.data = pd.DataFrame(data)
        self.__patient = 1
        self.__max_patient = patient
        if store_path.endswith('.csv'):
            if (follow_write or rewrite) and len(glob(store_path[:-4] + '*')) == 1:
                self.__store_path = glob(store_path[:-4] + '*')[0]
            else:
                self.__store_path = store_path[:-4] + '-' + time.strftime("%Y%m%d%H", time.localtime()) + '.csv'
        else:
            if (follow_write or rewrite) and len(glob(store_path + '*')) == 1:
                self.__store_path = glob(store_path + '*')[0]
            else:
                self.__store_path = store_path + '-' + time.strftime("%Y%m%d%H", time.localtime()) + '.csv'

        if follow_write and os.path.exists(self.__store_path):
            self.data = pd.read_csv(self.__store_path, index_col=0, header=0)

    def AddOne(self, case_name, content):
        if isinstance(content, dict):
            one_df = pd.DataFrame(content, index=[case_name])
        elif isinstance(content, list):
            one_df = pd.DataFrame(dict(zip(list(self.data.columns), content)), index=[case_name])
        elif isinstance(content, pd.Series):
            one_df = pd.DataFrame(content).T

        self.data = pd.concat((self.data, one_df), axis=0, sort=False)
        if self.__patient >= self.__max_patient:
            self.Save()
            self.__patient = 1
        self.__patient += 1


    def Save(self):
        self.data.to_csv(self.__store_path)

class Eclog:
    def __init__(self, file):
        self.eclogger = logging.getLogger(file)
        self.eclogger.setLevel(level=logging.DEBUG)
        if not self.eclogger.handlers:
            # self.rotate_handler = RotatingFileHandler("FECA.log", maxBytes=1024 * 1024, backupCount=5)
            self.rotate_handler = RotatingFileHandler(file, maxBytes=1024 * 1024, backupCount=5)
            self.rotate_handler.setLevel(level=logging.DEBUG)
            DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
            formatter = logging.Formatter(fmt='%(asctime)s(File:%(name)s,Line:%(lineno)d, %(funcName)s) - %(levelname)s - %(message)s', datefmt=DATE_FORMAT)

            self.rotate_handler.setFormatter(formatter)
            self.eclogger.addHandler(self.rotate_handler)

    def GetLogger(self):
        return self.eclogger


