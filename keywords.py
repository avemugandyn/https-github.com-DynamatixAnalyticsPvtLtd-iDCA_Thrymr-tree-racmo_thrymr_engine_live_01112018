
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

class Keywords(Base):
    __tablename__ = 'keywords'
    __table_args__ = {'autoload': True}
    
class SuspendKeywords(Base):
    __tablename__ = 'suspend_keywords'
    __table_args__ = {'autoload': True}


def loadSession():
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

session = loadSession()

def create_link_between_keyword(kdf):
    for i,row in kdf.iterrows():
        if row['sub'] != []:
            k = session.query(Keywords).filter(Keywords.keyword == json.dumps(row['keyword'])).first()
            link_k_ids = []
            for lk in row['sub']:
                kk = session.query(Keywords).filter(Keywords.keyword ==json.dumps(kdf[kdf['keywordHash']==lk]['keyword'].values[0])).first()
                if kk != None:
                    link_k_ids.append(kk.id)
            if k != None:
                k.sub = json.dumps(link_k_ids)
                session.commit()


def insert_record_in_keyword():
    with open('Keyword.pickle', 'rb') as handle:
        b=pickle.load(handle,encoding='latin1')
       
    kxddf=b['keywords']
    suspkdf=b['susKeyword']
    # # copy1 postgres copy2 pydblite
    print(kxddf)
    for i,r in kxddf.iterrows():
        
        k = Keywords(file_class =r['fileclass'], file_type =r['filetype'], purpose=r['purpose'],
                     decision_type=r['decision_type'], keyword=json.dumps(r['keyword']),bias=r['bias'])
        session.add(k)
        session.commit()
    # print(len(suspkdf),suspkdf)
    for j,sr in suspkdf.iterrows():
        Sk = SuspendKeywords(file_class=sr['file_class'], file_type=sr['file_type'],
             keyword=json.dumps(sr['keyword']),remove_class=sr['remove_class'])
        session.add(Sk)
        session.commit()
        
    return kxddf
            
if __name__ == '__main__':
    kdf = insert_record_in_keyword()
    create_link_between_keyword(kdf)

    
