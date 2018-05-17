from configuration.configuration import ConfigClass
import pickle

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
engine = create_engine(ConfigClass.SQLALCHEMY_DATABASE_URI, client_encoding="UTF-8")
Base = declarative_base(engine)

class Keywords(Base):
    __tablename__ = 'keywords'
    __table_args__ = {'autoload': True}

class File_class(Base):
    __tablename__ = 'file_class'
    __table_args__ = {'autoload': True}

class File_types(Base):
    __tablename__ = 'file_types'
    __table_args__ = {'autoload': True}

class Purpose(Base):
    __tablename__ = 'purpose'
    __table_args__ = {'autoload': True}

def loadSession():
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

session = loadSession()

def insert_record_in_keyword():
    with open('../Final_Keyword_Analysis.pickle', 'rb') as handle:
        b = pickle.load(handle)
    kxddf = b['keywordsX']
    kxddf.drop(173, inplace=True)
    print(kxddf.columns)
    # copy1 postgres copy2 pydblite
    for i,r in kxddf.iterrows():
        file_clss = session.query(File_class).filter(File_class.file_class_name ==r['fileclass']).first()
        file_typ = session.query(File_types).filter(File_types.type == r['filetype']).first()
        purpose = session.query(Purpose).filter(Purpose.purpose_type == r['purpose']).first()
        bias = session.query(File_class).filter(File_class.file_class_name == r['fileclass']).first()

        k = Keywords(file_class_id=file_clss.id,file_type_id=file_typ.id,purpose_id=purpose.id,
                     decision_type=r['notification_type'],keyword_list=json.dumps(r['keyword']),bias=bias.id)
        session.add(k)

        session.commit()

insert_record_in_keyword()
