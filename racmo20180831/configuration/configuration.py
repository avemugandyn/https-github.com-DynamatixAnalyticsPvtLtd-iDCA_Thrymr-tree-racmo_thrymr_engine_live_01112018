import os
from pymongo import MongoClient

class DbConf(object):
    name = "racmo_interface"
    username = "racmo"    #os.environ.get('DB_USER')
    password = "test123" #os.environ.get('DB_PASSWORD') 
    port = "3306"
    host = "localhost"
    # connect to Mongo-database
    client = MongoClient('mongodb://localhost:27017/')
    try:
        mdb = client.racmo
        fileData = mdb.fileData
    except Exception as e:
        print(e)


class ConfigClass(object):
    SECRET_KEY = os.getenv('SECRET_KEY', '_5y2LF4Q8z\n\xce')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL','mysql+pymysql://'+DbConf.username+':'+\
                                        DbConf.password+'@'+DbConf.host+':'+DbConf.port+'/'+ DbConf.name)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    NEW_FOLDER = '/home/racmo/engine/upload'  
    ARCHIVE_FOLDER = '/home/racmo/engine/archived'


class SessionConf(object):
    TIMEOUT = 20


