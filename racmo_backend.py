import os
import time
import json
import ast
import datetime
import pickle
import hashlib
from os import listdir
from os.path import isfile, join, isdir
from math import floor
from pydblite import Base
import PIL
from PIL import Image
import numpy as np
from shapely.geometry import box
from shapely.ops import cascaded_union
from pymongo import MongoClient
from gridfs import GridFS
from bson import ObjectId
from shutil import copyfile
from flask import Flask, request, jsonify, send_from_directory, session, g, redirect, send_from_directory, flash
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import pdfquery
from flask_login import login_user , logout_user , current_user , login_required
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from functools import wraps

decisionclass={'N1':'ADMISSION','N2':'TRANSFER OF LEGAL REPRESENTATION PERMITTED',
           'N3':'ENFORCEMENT ORDERS','N4':'STATEMENT OF PAYMENT',
           'N5':'REJECTED CLAIMS','N6':'REJECTED TRANSFER OF LEGAL REPRESENTATION',
           'N7':'HEARING','N8':'ASSET INQUIRY',
           'N9':'PLACE OF RESIDENCY REQUIRED',
           'N10':'REQUIREMENT'}

class dbConf(object):
    name = "racmo_one"
    username = "postgres"
    password = "test"
    port = "5432"
    host = "localhost"
    # connect to Mongo-database
    client = MongoClient('mongodb://localhost:27017/')
    try:
        mdb = client.racmo
        Notifications = mdb.Notifications
        PdfFiles=mdb.PdfFiles
        Keywords=mdb.Keywords
    except Exception as e:
        print(e)

class ConfigClass(object):
    SECRET_KEY = os.getenv('SECRET_KEY','THIS IS AN INSECURE SECRET')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL','postgresql://'+dbConf.username+':'+dbConf.password+'@'+dbConf.host+':'+dbConf.port+'/'+dbConf.name)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = "/home/thrymr/Racmo/pro"
    PDF_DIR = "/home/thrymr/Racmo/RacmoIT/process/Gestured documents 15-02-2018"
    


with open(ConfigClass.UPLOAD_FOLDER+'/TestFileGroup.pickle', 'rb') as handle:
    a=pickle.load(handle,encoding='latin1')
fgdf=a['fg']

fgdf=fgdf.fillna('')

def get_pgnum(filename):
    pdf=pdfquery.PDFQuery(ConfigClass.UPLOAD_FOLDER+"/"+filename)
    pdf.load()
    pgn=len(pdf.tree.getroot().getchildren())
    return pgn

# def insert_records(filename,org_filename):
#     mdb = dbConf.mdb
#     Notifications = dbConf.Notifications
#     # read in the image.
#     datafile = open(ConfigClass.UPLOAD_FOLDER+"/"+filename, "rb");
#     thedata = datafile.read()
#     # fs = GridFS(mdb)
#     # stored = fs.put(thedata, filename="testimage")
#     noti = {
#         "filename": org_filename,
#         "file_text":get_all_text(filename,org_filename),
#         "type": get_file_type(filename)
#     }
#     Notifications.insert_one(noti)
#     return

def get_all_filegroups():
	return fgdf[['filegroup','group_predicted_class']].rename(columns={"group_predicted_class":"decision"}).to_dict('records')

def get_filegroup_data(filegroup):
    for i , r in fgdf[fgdf['filegroup']==filegroup].iterrows():
        return dict(r)
def save_inputfeedback(js,filegroup):
        fgdf.loc[fgdf['filegroup']==filegroup,'decision(Actual)']=js['decision']
        fgdf.loc[fgdf['filegroup']==filegroup,'Court(Actual)']=js['court']
        fgdf.loc[fgdf['filegroup']==filegroup,'Solictor(Actual)']=js['solicitor']
        fgdf.loc[fgdf['filegroup']==filegroup,'Document date(Actual)']=js['docDate']
        fgdf.loc[fgdf['filegroup']==filegroup,'Stamp date(Actual)']=js['stampDate']
        fgdf.loc[fgdf['filegroup']==filegroup,'Procedure_Type(Actual)']=js['procedure']
        fgdf.loc[fgdf['filegroup']==filegroup,'Amount(Actual)']=js['amount']
        fgdf.loc[fgdf['filegroup']==filegroup,'Auto(Actual)']=js['auto']
        fgdf.loc[fgdf['filegroup']==filegroup,'Time Frame(Actual)']=js['days']
        a = {'fg':fgdf}
        with open('/home/thrymr/Racmo/pro/TestFileGroup.pickle', 'wb') as handle:
            pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_app(test_config=None):
    login_manager = LoginManager()
    app = Flask(__name__)
    app.config.from_object(__name__+'.ConfigClass')
    db = SQLAlchemy(app)
    
    class User(db.Model):
        __tablename__ = "users"
        id = db.Column('user_id',db.Integer , primary_key=True)
        username = db.Column('username', db.String(20), unique=True , index=True)
        password = db.Column('password' , db.String(10))
        token = db.Column('token' , db.String(100))
        email = db.Column('email',db.String(50),unique=True , index=True)
        registered_on = db.Column('registered_on' , db.DateTime)

        def __init__(self , username ,password , email,registered_on, token):
            self.username = username
            self.password = password
            self.token = token
            self.email = email
            self.registered_on = registered_on

        def is_authenticated(self):
            return True

        def is_active(self):
            return True

        def is_anonymous(self):
            return False

        def get_id(self):
            return self.id
        
        def get(user_id):
            return User.query.filter(User.id== user_id).first()
        

    db.create_all()
    
    if not User.query.filter(User.email=='user@abc.com').first():
        user = User(email='user@abc.com', username='user', password="User@123", registered_on=datetime.datetime.utcnow(), token="test")
        db.session.add(user)
        db.session.commit()

    if not User.query.filter(User.email=='admin@abc.com').first():
        user1= User(email='admin@abc.com', username='admin', password="Admin@123", registered_on=datetime.datetime.utcnow(), token="test")
        db.session.add(user1)
        db.session.commit()

    @app.before_request
    def before_request():
        session.permanent = True
        app.permanent_session_lifetime = datetime.timedelta(minutes=10)
        session.modified = True
        g.user = current_user
        
    # @app.after_request
    # def after_request(response):
    #     response.headers.add('Access-Control-Allow-Origin', '*')
    #     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    #     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    #     return response

    def check_token(user_access_token):
        user = User.query.filter_by(token=user_access_token).first()
        if user:
            return True
        return False
    
    
#     @login_manager.request_loader
#     def load_user_from_request(request):

#         # first, try to login using the api_key url arg
#         api_key = request.args.get('api_key')
#         if api_key:
#             user = User.query.filter_by(api_key=api_key).first()
#             if user:
#                 return user

#         # next, try to login using Basic Auth
#         api_key = request.headers.get('Authorization')
#         if api_key:
#             api_key = api_key.replace('Basic ', '', 1)
#             try:
#                 api_key = base64.b64decode(api_key)
#             except TypeError:
#                 pass
#             user = User.query.filter_by(api_key=api_key).first()
#             if user:
#                 return user

#         # finally, return None if both methods did not login the user
#         return None
    
#     @login_manager.user_loader
#     def load_user(user_id):
#         return User.get(user_id)

    @app.route("/api", methods=["POST"])
    def hello():
        data = request.get_json()
        x_auth = data['xauth']
        if check_token(x_auth):
            js = request.get_json(silent=True)
            print(js)
            response=jsonify({"js":js,"server":True})
            return response
        resp = jsonify({"ErrorMessage": " 'User not logged In' "})
        resp.status_code = 204
        return resp
    
    @app.route("/api/get-all-filegroups",methods=['POST'])
    @cross_origin(origins="*")
    def viewmorepdfs():
        data = request.get_json()
        x_auth = data['xauth']
        print(session, data)
        if check_token(x_auth):
            response=jsonify({"filegroups":get_all_filegroups()})
            # response.headers['Access-Control-Allow-Origin']= '*'
            # response.headers['Access-Control-Allow-Headers']= 'Content-Type,Authorization'
            # response.headers['Access-Control-Allow-Methods']= 'GET,PUT,POST,DELETE'
            return response
        resp = jsonify({"ErrorMessage": " 'User not logged In' "})
        resp.status_code = 204
        # response.headers['Access-Control-Allow-Origin']= '*'
        # response.headers['Access-Control-Allow-Headers']= 'Content-Type,Authorization'
        # response.headers['Access-Control-Allow-Methods']= 'GET,PUT,POST,DELETE'
        return resp

        # send_file("outputname-01.png", mimetype='image/gif',add_etags=false)
        # return "failure" 
 
    @app.route("/api/get-file-groupdata", methods=["POST"])
    def pdfimage():
        data = request.get_json()
        x_auth = data['xauth']
        filegroup=data['filegroup']
        print("inside pdfimage",current_user,session, data)
        if check_token(x_auth):
            fg=get_filegroup_data(filegroup)
            dec=''
            if fg['group_predicted_class']=='N2+N4':
                dec='N2 '+decisionclass['N2']+'+'+'N4 '+decisionclass['N4']
            
            elif fg['group_predicted_class']!='':
                dec=fg['group_predicted_class']+' '+decisionclass[fg['group_predicted_class']]
            fls=[]
            print (fg)
            for fl in fg['files']:
                ls=[]
                filename = secure_filename(fl)
                try:
                   
                    copyfile(join(ConfigClass.PDF_DIR,fl), join(ConfigClass.UPLOAD_FOLDER,filename))
                    print("cd "+ConfigClass.UPLOAD_FOLDER+" && pdftoppm "+filename+" main -png")
                    os.system("cd "+ConfigClass.UPLOAD_FOLDER+" && pdftoppm \""+filename+"\" main -png")
                    
                    
                    pgnum=get_pgnum(filename)
                    for i in range(0,pgnum):
                        if(pgnum>10 and i+1<10):
                            ls.append(ConfigClass.UPLOAD_FOLDER+"/main-0"+str(i+1)+".png")
                        else:
                            ls.append(ConfigClass.UPLOAD_FOLDER+"/main-"+str(i+1)+".png")
                    imgs    = [ PIL.Image.open(i) for i in ls ]
                    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
                    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
                    imgs_comb = PIL.Image.fromarray( imgs_comb)
                    imgs_comb.save( ConfigClass.UPLOAD_FOLDER+"/"+filename.rsplit('.',1)[0]+'.jpg')
                    fls.append({'filename':filename,'uploadfilename':filename.rsplit('.',1)[0]+'.jpg?'+str(time.time()),'Type':fg['filetypes'][fg['files'].index(fl)]})
                except Exception as e:
                     fls.append({'filename':filename,'uploadfilename':'unparsed.png','Type':fg['filetypes'][fg['files'].index(fl)]})
                   
            pred={'decision':dec,'auto':fg['Auto'],'procedure':fg['Procedure_Type'],'court':fg['Court'],
                              'solicitor':fg['Solictor'],'amount':fg['Amount'],'days':fg['MIN DAYS'],'docDate':fg["Document date"],'stampDate':fg["Stamp date"],'keywords':str(fg['Keywords'])}
            real=pred
            realList=['decision(Actual)','Auto(Actual)','Procedure_Type(Actual)','Court(Actual)','Solictor(Actual)','Amount(Actual)','Time Frame(Actual)','Document date(Actual)','Stamp date(Actual)']
            f=True
            for r in realList:
                if not r in fg.keys():
                    f=False
                
            if f:
                real={'decision':fg['decision(Actual)'],'auto':fg['Auto(Actual)'],'procedure':fg['Procedure_Type(Actual)'],'court':fg['Court(Actual)'],
                              'solicitor':fg['Solictor(Actual)'],'amount':fg['Amount(Actual)'],'days':fg['Time Frame(Actual)'],'docDate':fg["Document date(Actual)"],'stampDate':fg["Stamp date(Actual)"],'keywords':str(fg['Keywords'])}
                
                
            response=jsonify({'files':fls,
                "pred":pred,
                "real":real,
                })

            return response
        resp = jsonify({"ErrorMessage": " 'User not logged In' "})
        resp.status_code = 204
        return resp
       
    @app.route('/api/upload/<path:x_auth>/<path:filename>',methods=["GET"])
    def upload_file(x_auth,filename):
        
        if check_token(x_auth):
            return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
        resp = jsonify({"ErrorMessage": " 'User not logged In' "})
        resp.status_code = 204
        return resp
    
    @app.route('/api/input-feedback/<path:filegroup>',methods=['GET','POST'])
    def receive_feedback(filegroup):
        print("her22222e")
        data = request.get_json()
        x_auth = data['xauth']
        if check_token(x_auth):
            js = request.get_json(silent=True)
            save_inputfeedback(js,filegroup)
            return "success"
        resp = jsonify({"ErrorMessage": " 'User not logged In' "})
        resp.status_code = 204
        return resp
    
    
    @app.route('/api/login',methods=['POST'])
    #@cross_origin()
    def login():
        print(request.get_json())
        login_data = request.get_json()
        username = login_data['email']
        password = login_data['password']
        registered_user = User.query.filter_by(username=username,password=password).first()
        if registered_user is None: 
            resp = jsonify({"ErrorMessage": " 'User not there' "})
            resp.status_code = 204
            return resp
        print("Before==",session)
        login_user(registered_user,remember=True)
        #token = g.user.generate_auth_token()
        session['api_session_token'] = hashlib.sha1((session['_id']+str(time.time())).encode()).hexdigest()
        print("After-",session)
        resp = jsonify({"Message": " 'User logged In' ","token":session['api_session_token']})
        resp.status_code = 200
        registered_user.token = session['api_session_token']
        db.session.add(registered_user)
        db.session.commit()
        # resp.headers.add('Access-Control-Allow-Origin', '*')
        # resp.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        # resp.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
        return resp

    @app.route('/api/logout')
    #@login_required
    def logout():
        session={}
        print("In logout==",session)

        logout_user()
        print("After logout--",session)
        resp = jsonify({"Message": " 'User logged Out' "})
        resp.status_code = 200
        return resp
    
    @app.route('/')
    def index():
        return "Hello RACMO"
    
    #CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    #CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    CORS(app, resources=r'/api/*', headers='Content-Type')
    login_manager.init_app(app)
    login_manager.login_view = 'api.login'
    app.config['CORS_HEADERS'] = 'Content-Type'
    return app


app = create_app()
if __name__=='__main__':
    app.run(port=5052,debug=False, host="0.0.0.0")

    



