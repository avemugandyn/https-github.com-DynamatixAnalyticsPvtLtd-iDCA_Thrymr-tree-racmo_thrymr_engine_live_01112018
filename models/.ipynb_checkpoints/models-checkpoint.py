from flask_sqlalchemy import SQLAlchemy
from collections import OrderedDict

class Models(object):

    def __init__(self, app):
        self.app = app
        self.db = SQLAlchemy(app)
        self.FileClass, self.Keyword, self.UserDetails,\
        self.FileClassificationResult, self.SuspendKeywords, self.FileGroup= self.model_classes(self.db)

    @staticmethod
    def model_classes(db):

        class FileClass(db.Model):
            __tablename__ = 'file_class'
            id = db.Column(db.Integer, primary_key=True)
            file_class_name = db.Column(db.Text, nullable=False)
            decision = db.Column(db.Text, nullable=False)
            col_new = db.Column(db.Text, nullable=True)

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
            __tablename__='file_classification'
            id = db.Column(db.Integer, primary_key=True)
            file_name = db.Column(db.Text, nullable=True)
            file_type = db.Column(db.Text, nullable=True)
            file_group = db.Column(db.Text, nullable=True)
            predicted_classes = db.Column(db.Text, nullable=True)

        class FileGroup(db.Model):
            __tablename__= 'filegroup_extraction'
            id = db.Column(db.Integer, primary_key=True)
            file_group = db.Column(db.Text, nullable=True)
            court = db.Column(db.Text, nullable=True)
            solicitor = db.Column(db.Text, nullable=True)
            procedure_type = db.Column(db.Text, nullable=True)
            time_frame = db.Column(db.Text, nullable=True)
            document_date = db.Column(db.Text, nullable=True)
            stamp_date = db.Column(db.Text, nullable=True)
            auto = db.Column(db.Text, nullable=True)
            amount = db.Column(db.Text, nullable=True)
            date_of_hearing = db.Column(db.Text, nullable=True)
            debtor = db.Column(db.Text, nullable=True)
        
        class UserDetails(db.Model):
            __tablename__='user_details'
            id = db.Column(db.Integer, primary_key=True)
            name = db.Column(db.Text, nullable=True)
            new_folder = db.Column(db.Text, nullable=True)
            archive_folder = db.Column(db.Text, nullable=True)

        db.create_all()

        #enum fileclass
        fclass=OrderedDict([('N1','ADMISSION'),('N2','TRANSFER OF LEGAL REPRESENTATION PERMITTED'),
           ('N3','ENFORCEMENT ORDERS'),('N4','STATEMENT OF PAYMENT'),
           ('N5','REJECTED CLAIMS'),('N6','REJECTED TRANSFER OF LEGAL REPRESENTATION'),
           ('N7','HEARING'),('N8','ASSET INQUIRY'),
           ('N9','PLACE OF RESIDENCY REQUIRED'),
           ('N10','REQUIREMENT'),('N11','BANK TRANSFERS'),('N12','SOLICITOR REMOVAL'),('N13','SUSPENDED HEARINGS'),
            ('N14','CANCELLED HEARINGS'),('N15','NEGATIVE ASSET INQUIRY'),('N16','PENDING ASSET INQUIRY')])

        for k,v in fclass.items():
            new_type = FileClass(file_class_name=k,decision=v)
            db.session.add(new_type)
            db.session.commit()
        
        if not UserDetails.query.filter(UserDetails.name == 'Testuser').first():
            new_user = UserDetails(name='Testuser',
                               new_folder='/home/thrymr/Notifications/test_user/new',
                               archive_folder='/home/thrymr/Notifications/test_user/archived')
            db.session.add(new_user)
            db.session.commit()

        return  FileClass, Keyword, UserDetails, FileClassificationResult, SuspendKeywords, FileGroup
