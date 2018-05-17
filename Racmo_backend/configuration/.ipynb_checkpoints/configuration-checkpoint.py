import os
from pymongo import MongoClient
from pydblite.pydblite import Base


class DbConf(object):
    name = "test_racmo_one"
    username = "postgres"
    password = "test"
    port = "5432"
    host = "localhost"
    # connect to Mongo-database
    client = MongoClient('mongodb://localhost:27017/')
    try:
        mdb = client.racmo
        fileData = mdb.fileData
    except Exception as e:
        print(e)

#Mongo class definitions
# emp = { "id":"","filename":"",
#         "image_file":"",
#         "actualfile":""
#        }


class ConfigClass(object):
    SECRET_KEY = os.getenv('SECRET_KEY', 'THIS IS AN INSECURE SECRET')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL','postgresql://' + DbConf.username + ':' + DbConf.password + '@' + DbConf.host + ':' + DbConf.port + '/' + DbConf.name)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = "/home/thrymr/Racmo/pro"
    PDF_DIR = "/home/thrymr/Racmo/RacmoIT/process/Gestured documents 15-02-2018"


class SessionConf(object):
    TIMEOUT = 20


class PyDbLite(object):
    pyDbLite_db = Base('pyDbLite_db')
    pyDbLite_db.create('user_info','keyword_df','file_filegroup','result_df', mode="open")


