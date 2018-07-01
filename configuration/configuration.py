import os
from pymongo import MongoClient

class DbConf(object):
    name = "aim"
    username = "racmo2"    #os.environ.get('DB_USER')
    password = "7N8IswX26@18" #os.environ.get('DB_PASSWORD') 
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
    NEW_FOLDER = '/home/thrymr/Notification Engine/new'
    ARCHIVE_FOLDER = '/home/thrymr/Notification Engine/archived'


class SessionConf(object):
    TIMEOUT = 20


