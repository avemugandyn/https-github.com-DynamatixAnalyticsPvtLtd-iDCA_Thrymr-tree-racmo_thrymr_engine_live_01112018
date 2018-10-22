from flask_sqlalchemy import SQLAlchemy
from collections import OrderedDict
import datetime
class Models(object):

    def __init__(self, app):
        self.app = app
        self.db = SQLAlchemy(app)
        self.Keyword, self.FileClassificationResult, self.SuspendKeywords,\
            self.FileGroup, self.ProccessLog = self.model_classes(self.db)

    @staticmethod
    def model_classes(db):

        class Keyword(db.Model):
            __tablename__ = 'keywords'
            id = db.Column(db.Integer, primary_key=True)
            file_class= db.Column(db.Text,nullable=True)
            file_type = db.Column(db.Text, nullable=True)
            purpose = db.Column(db.Text, nullable=True)
            decision_type = db.Column(db.Text, nullable=False)
            keyword = db.Column(db.Text, nullable=False)
            bias = db.Column(db.Text, nullable=True)
            sub = db.Column(db.Text, nullable=True)
            
        class SuspendKeywords(db.Model):
            __tablename__ = 'suspend_keywords'
            id = db.Column(db.Integer, primary_key=True) 
            file_class= db.Column(db.Text,nullable=True)
            file_type = db.Column(db.Text, nullable=True)
            keyword = db.Column(db.Text, nullable=False)
            remove_class = db.Column(db.Text, nullable=False)

        class FileClassificationResult(db.Model):
            __tablename__='file_classification_staging'
            id = db.Column(db.Integer, primary_key=True)
            file_name = db.Column(db.Text, nullable=True)
            lexnet_filename = db.Column(db.Text, nullable=True)
            file_type = db.Column(db.Text, nullable=True)
            file_group = db.Column(db.Text, nullable=True)
            keyword = db.Column(db.Text, nullable=True)
            predicted_classes = db.Column(db.Text, nullable=True)
            engine_comments = db.Column(db.Text,nullable=True)
            batch_id = db.Column(db.Integer, nullable = False)
            last_updated_by =  db.Column(db.Integer, nullable = False, default=1)
            creation_date = db.Column(db.Date, nullable = False) #default now()

        class FileGroup(db.Model):
            __tablename__= 'filegroup_extraction_staging'
            file_group_id = db.Column(db.Integer, primary_key=True)
            file_group = db.Column(db.Text, nullable=True)
            court = db.Column(db.Text, nullable=True)
            solicitor = db.Column(db.Text, nullable=True)
            procedure_type = db.Column(db.Text, nullable=True)
            time_frame = db.Column(db.Text, nullable=True)
            document_date = db.Column(db.Text, nullable=True)
            document_date_initial = db.Column(db.Text, nullable=True)
            stamp_date = db.Column(db.Text, nullable=True)
            stamp_date_initial = db.Column(db.Text, nullable=True)
            auto = db.Column(db.Text, nullable=True)
            amount = db.Column(db.Text, nullable=True)
            possible_amount = db.Column(db.Text, nullable=True)
            amount_initial = db.Column(db.Text, nullable=True)
            date_of_hearing = db.Column(db.Text, nullable=True)
            date_of_hearing_initial = db.Column(db.Text, nullable=True)
            debtor = db.Column(db.Text, nullable=True)
            debtor_initial = db.Column(db.Text, nullable=True)
            possible_debtors = db.Column(db.Text, nullable=True)
            batch_id = db.Column(db.Integer, nullable = False)
            court_number = db.Column(db.Text, nullable=True)
            court_number_initial = db.Column(db.Text, nullable=True)
            court_initial = db.Column(db.Text, nullable=True)
            procedure_type_initial = db.Column(db.Text, nullable=True)
            solicitor_initial = db.Column(db.Text, nullable=True)
            auto_initial = db.Column(db.Text, nullable=True)
            creation_date = db.Column(db.Date, nullable = False) #default now()
            last_update_date = db.Column(db.Date, nullable = False,default=datetime.datetime.now())
            
        class ProccessLog(db.Model):
            __tablename__ = 'process_log'
            batch_id = db.Column(db.Integer, primary_key=True)
            process_date = db.Column(db.Date, nullable = False)
            creation_date = db.Column(db.Date, nullable = False)
            created_by = db.Column(db.Integer, nullable = False, default=1)
            last_update_date = db.Column(db.Date, nullable = False, default=datetime.datetime.now())
            last_updated_by =  db.Column(db.Integer, nullable = False, default=1)
            engine_running = db.Column(db.Integer, nullable = False, default=1)

        db.create_all()

        return Keyword, FileClassificationResult, SuspendKeywords, FileGroup, ProccessLog

    
           
