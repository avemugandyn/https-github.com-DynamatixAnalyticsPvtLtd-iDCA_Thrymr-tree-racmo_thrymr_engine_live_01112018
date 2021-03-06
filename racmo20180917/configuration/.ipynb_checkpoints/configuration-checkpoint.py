import os
from pymongo import MongoClient

class DbConf(object):
    name = "test_db"
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
    UPLOAD_FOLDER = "/home/thrymr/Racmo/pro"
    PDF_DIR = "/home/thrymr/Racmo/RacmoIT/process/Gestured documents 15-02-2018"
    NEW_FOLDER = '/home/thrymr/Notifications/test_user/new'
    ARCHIVE_FOLDER = '/home/thrymr/Notifications/test_user/archived'


class SessionConf(object):
    TIMEOUT = 20


