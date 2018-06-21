from configuration.configuration import ConfigClass
import pickle
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import hashlib
import unidecode

engine = create_engine(ConfigClass.SQLALCHEMY_DATABASE_URI)
Base = declarative_base(engine)

def loadSession():
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

session = loadSession()

class FileClassification(Base):
    __tablename__ = 'file_classification'
    __table_args__ = {'autoload': True}
    
class FileGroupExtraction(Base):
    __tablename__ = 'filegroup_extraction'
    __table_args__ = {'autoload': True}
with open('../SampleData.pickle', 'rb') as handle:
        b=pickle.load(handle,encoding='latin1')
fdf=b['fdf']
fgdf=b['fg']
for i , r in fdf.iterrows():
    k = FileClassification(file_name =r['filename'], file_group =r['filegroup'], file_type=r['filetype'],
                     predicted_classes=json.dumps(['final_categ']))
    session.add(k)
    session.commit()

for i , r in fgdf.iterrows():
    k = FileGroupExtraction(file_group = r['filegroup'], court = r['Court'],
            solicitor = r['Solictor'],
            procedure_type = r['Procedure_Type'],
            time_frame = r['Time Frame'],
            document_date = r['Document date'],
            stamp_date = r['Stamp date'],
            auto = r['Auto'],
            amount = r['Amount'],
            date_of_hearing =r['Date_of_hearing'],
            debtor = r['Debtor'])
    session.add(k)
    session.commit()