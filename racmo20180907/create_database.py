from sqlalchemy import create_engine
from configuration.configuration import DbConf,ConfigClass
if __name__ == '__main__':
    engine = create_engine(ConfigClass.SQLALCHEMY_DATABASE_URI)
    engine.execute("CREATE DATABASE IF NOT EXISTS test_db")