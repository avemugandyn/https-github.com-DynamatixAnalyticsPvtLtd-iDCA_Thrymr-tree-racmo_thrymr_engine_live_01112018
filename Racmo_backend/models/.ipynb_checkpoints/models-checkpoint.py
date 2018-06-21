from flask_sqlalchemy import SQLAlchemy
from collections import OrderedDict
class Models(object):

    def __init__(self, app):
        self.app = app
        self.db = SQLAlchemy(app)
        self.User, self.FileType, self.Purpose, self.FileClass, self.Keyword,\
        self.FileGroup, self.FileInfo = self.model_classes(self.db)

    @staticmethod
    def model_classes(db):

        class User(db.Model):
            __tablename__ = 'users'
            id = db.Column(db.Integer, primary_key=True)
            public_id = db.Column(db.String(50))  # use for token
            name = db.Column(db.String(50))
            password = db.Column(db.String(80))
            admin = db.Column(db.Boolean)

        class FileType(db.Model):
            __tablename__ = 'file_types'
            id = db.Column(db.Integer, primary_key=True)
            type = db.Column(db.String(100), nullable=False)

        class Purpose(db.Model):
            __tablename__ = 'purpose'
            id = db.Column(db.Integer, primary_key=True)
            purpose_type = db.Column(db.Text, nullable=False)

        class FileClass(db.Model):
            __tablename__ = 'file_class'
            id = db.Column(db.Integer, primary_key=True)
            file_class_name = db.Column(db.Text, nullable=False)
            decision = db.Column(db.Text, nullable=False)

        class Keyword(db.Model):
            __tablename__ = 'keywords'
            id = db.Column(db.Integer, primary_key=True)
            file_class_id = db.Column(db.Integer, db.ForeignKey("file_class.id"), nullable=True)
            file_class = db.relationship("FileClass", lazy="joined", foreign_keys=[file_class_id],
                                         backref=db.backref("keywords", cascade="all, delete-orphan"))

            file_type_id = db.Column(db.Integer, db.ForeignKey("file_types.id"), nullable=True)
            file_type = db.relationship("FileType", lazy="joined", foreign_keys=[file_type_id],
                                        backref=db.backref("keywords", cascade="all, delete-orphan"))

            purpose_id = db.Column(db.Integer, db.ForeignKey("purpose.id"), nullable=True)
            purpose = db.relationship("Purpose", lazy="joined", foreign_keys=[purpose_id],
                                      backref=db.backref("keywords", cascade="all, delete-orphan"))

            decision_type = db.Column(db.Text, nullable=False)
            keyword_list = db.Column(db.Text, nullable=False)
            bias = db.Column(db.Integer, db.ForeignKey("file_class.id"), nullable=True)
            bias_map = db.relationship("FileClass", lazy="joined", foreign_keys=[bias])

        class FileGroup(db.Model):
            __tablename__ = 'file_group'
            id = db.Column(db.Integer, primary_key=True)
            file_group_name = db.Column(db.Text, nullable=False)
            predicted_class = db.Column(db.Integer, db.ForeignKey("file_class.id"), nullable=True)
            file_class = db.relationship("FileClass", lazy="joined", foreign_keys=[predicted_class],
                                         backref=db.backref("file_group", cascade="all, delete-orphan"))

        class FileInfo(db.Model):
            __tablename__ = 'file_info'
            id = db.Column(db.Integer, primary_key=True)
            file_group_id = db.Column(db.Integer, db.ForeignKey("file_group.id"), nullable=True)
            file_group = db.relationship("FileGroup", lazy="joined", foreign_keys=[file_group_id],
                                         backref=db.backref("file_info", cascade="all, delete-orphan"))
            file_data_id = db.Column(db.Text, nullable=False)
            file_name = db.Column(db.Text, nullable=False)
        db.create_all()

        if not FileType.query.filter(FileType.type == 'NOTIFICATION').first():
            new_type = FileType(type='NOTIFICATION')
            db.session.add(new_type)
            db.session.commit()
        if not FileType.query.filter(FileType.type == 'TICKET').first():
            new_type = FileType(type='TICKET')
            db.session.add(new_type)
            db.session.commit()
        if not FileType.query.filter(FileType.type == 'OTHER').first():
            new_type = FileType(type='OTHER')
            db.session.add(new_type)
            db.session.commit()
        if not FileType.query.filter(FileType.type == 'ANNEXURE').first():
            new_type = FileType(type='ANNEXURE')
            db.session.add(new_type)
            db.session.commit()
        #enum Purpose data
        if not Purpose.query.filter(Purpose.purpose_type == 'CLASSIFICATION').first():
            new_type = Purpose(purpose_type='CLASSIFICATION')
            db.session.add(new_type)
            db.session.commit()
        if not Purpose.query.filter(Purpose.purpose_type == 'EXTRACTION').first():
            new_type = Purpose(purpose_type='EXTRACTION')
            db.session.add(new_type)
            db.session.commit()
        #enum fileclass
        fclass=OrderedDict([('N1','ADMISSION'),('N2','TRANSFER OF LEGAL REPRESENTATION PERMITTED'),
           ('N3','ENFORCEMENT ORDERS'),('N4','STATEMENT OF PAYMENT'),
           ('N5','REJECTED CLAIMS'),('N6','REJECTED TRANSFER OF LEGAL REPRESENTATION'),
           ('N7','HEARING'),('N8','ASSET INQUIRY'),
           ('N9','PLACE OF RESIDENCY REQUIRED'),
           ('N10','REQUIREMENT'),('N11','BANK TRANSFERS')])
        for k,v in fclass.items():
            if not Purpose.query.filter(FileClass.file_class_name == k).first():
                new_type = FileClass(file_class_name=k,decision=v)
                db.session.add(new_type)
                db.session.commit()


        return User, FileType, Purpose, FileClass, Keyword, FileGroup, FileInfo
