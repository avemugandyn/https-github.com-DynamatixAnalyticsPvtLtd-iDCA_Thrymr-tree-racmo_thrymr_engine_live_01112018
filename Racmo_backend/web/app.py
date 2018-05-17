from flask import Flask, request, jsonify, make_response, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import jwt
import uuid
import time
import datetime
from functools import wraps
import os
from os.path import join
import pickle
import pdfquery
import PIL
from PIL import Image
import numpy as np
from shutil import copyfile
import io
from gridfs import GridFS
from bson import ObjectId
from bson.json_util import dumps
import base64
import json
# local packages
from models.models import Models
from configuration.configuration import DbConf,ConfigClass,PyDbLite
print("PyDbLite===>",PyDbLite)
decisionclass = {'N1': 'ADMISSION', 'N2': 'TRANSFER OF LEGAL REPRESENTATION PERMITTED',
                 'N3': 'ENFORCEMENT ORDERS', 'N4': 'STATEMENT OF PAYMENT',
                 'N5': 'REJECTED CLAIMS', 'N6': 'REJECTED TRANSFER OF LEGAL REPRESENTATION',
                 'N7': 'HEARING', 'N8': 'ASSET INQUIRY',
                 'N9': 'PLACE OF RESIDENCY REQUIRED',
                 'N10': 'REQUIREMENT'}


# with open(ConfigClass.UPLOAD_FOLDER + '/TestFileGroup.pickle', 'rb') as handle:
#     a = pickle.load(handle, encoding='latin1')
# fgdf = a['fg']
# fgdf = fgdf.fillna('')


def get_pgnum(filename):
    pdf = pdfquery.PDFQuery(ConfigClass.UPLOAD_FOLDER + "/" + filename)
    pdf.load()
    pgn = len(pdf.tree.getroot().getchildren())
    return pgn

def insert_in_postgres():
    Models.FileType.type = "Notifacation"
    Models.db.commit()

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
    pyDbLite_db = PyDbLite().pyDbLite_db
    # print(pyDbLite_db)
    # print("records==== ",pyDbLite_db.records.keys())
    # print("sddddddddddddddd"+str(pyDbLite_db.records[1]['result_df']))
    
    fgdf=pyDbLite_db.records[1]['result_df']
    fgdf = fgdf.fillna('')
    return fgdf[['filegroup', 'group_predicted_class']].rename(columns={"group_predicted_class": "decision"}).to_dict(
        'records')


def get_filegroup_data(filegroup):
    pyDbLite_db = PyDbLite().pyDbLite_db
    fgdf=pyDbLite_db.records[1]['result_df']
    fgdf = fgdf.fillna('')
    for i, r in fgdf[fgdf['filegroup'] == filegroup].iterrows():
        return dict(r)


def save_inputfeedback(js, filegroup):
    pyDbLite_db = PyDbLite().pyDbLite_db
    fgdf=pyDbLite_db.records[1]['result_df']

    fgdf.loc[fgdf['filegroup'] == filegroup, 'real_values'] = js
    # fgdf.loc[fgdf['filegroup'] == filegroup, 'decision(Actual)'] = js['decision']
    # fgdf.loc[fgdf['filegroup'] == filegroup, 'Court(Actual)'] = js['court']
    # fgdf.loc[fgdf['filegroup'] == filegroup, 'Solictor(Actual)'] = js['solicitor']
    # fgdf.loc[fgdf['filegroup'] == filegroup, 'Document date(Actual)'] = js['docDate']
    # fgdf.loc[fgdf['filegroup'] == filegroup, 'Stamp date(Actual)'] = js['stampDate']
    # fgdf.loc[fgdf['filegroup'] == filegroup, 'Procedure_Type(Actual)'] = js['procedure']
    # fgdf.loc[fgdf['filegroup'] == filegroup, 'Amount(Actual)'] = js['amount']
    # fgdf.loc[fgdf['filegroup'] == filegroup, 'Auto(Actual)'] = js['auto']
    # fgdf.loc[fgdf['filegroup'] == filegroup, 'Time Frame(Actual)'] = js['days']
    pyDb_id = pyDbLite_db.update(data,result_df=fgdf)


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(ConfigClass)
    model = Models(app)

    if not model.User.query.filter(model.User.name == 'Admin').first():
        hashed_password = generate_password_hash('Admin@123', method='sha256')
        new_user = model.User(public_id=str(uuid.uuid4()), name='Admin', password=hashed_password, admin=True)
        model.db.session.add(new_user)
        model.db.session.commit()
        
    if not model.User.query.filter(model.User.name == 'User1').first():
        hashed_password = generate_password_hash('User@1', method='sha256')
        new_user = model.User(public_id=str(uuid.uuid4()), name='User1', password=hashed_password, admin=False)
        model.db.session.add(new_user)
        model.db.session.commit()
        
    if not model.User.query.filter(model.User.name == 'User2').first():
        hashed_password = generate_password_hash('User@2', method='sha256')
        new_user = model.User(public_id=str(uuid.uuid4()), name='User2', password=hashed_password, admin=False)
        model.db.session.add(new_user)
        model.db.session.commit()

    def token_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = None
            if 'x-access-token' in request.headers:
                token = request.headers['x-access-token']
            if not token:
                return jsonify({'message': 'Token is missing!'}), 401
            try:
                data = jwt.decode(token, app.config['SECRET_KEY'])
                current_user = model.User.query.filter_by(public_id=data['public_id']).first()
            except:
                return jsonify({'message': 'Token is invalid!'}), 401
            if current_user is None:
                return jsonify({'message': 'Token is invalid!'}), 401

            return f(current_user, *args, **kwargs)

        return decorated

    @app.route('/user', methods=['POST'])
    @token_required
    def create_user(current_user):
        if not current_user.admin:
            return jsonify({'message': 'Cannot perform that function!'})

        data = request.get_json()

        hashed_password = generate_password_hash(data['password'], method='sha256')

        new_user = model.User(public_id=str(uuid.uuid4()), name=data['name'], password=hashed_password, admin=False)
        model.db.session.add(new_user)
        model.db.session.commit()

        return jsonify({'message': 'New user created!'})

    @app.route('/api/login', methods=['POST'])
    def login():
        auth = request.get_json()
        username = auth['email']
        password = auth['password']

        if not auth or not username or not password:
            return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})

        user = model.User.query.filter_by(name=username).first()

        if not user:
            return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})

        if check_password_hash(user.password, password):
            token = jwt.encode(
                {'public_id': user.public_id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=20)},
                app.config['SECRET_KEY'])

            return jsonify({'token': token.decode('UTF-8')})

        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})

    @app.route('/api/logout', methods=['PUT'])
    @token_required
    def logout(current_user):
        current_user.public_id = str(uuid.uuid4())
        model.db.session.commit()
        return jsonify({'message': "'Successfully Logout'"})

    @app.route("/api", methods=["POST"])
    @token_required
    def hello(current_user):
        js = request.get_json(silent=True)
        response = jsonify({"js": js, "server": True})
        return response

    @app.route("/api/get-all-filegroups", methods=['POST'])
    @cross_origin(origins="*")
    @token_required
    def viewmorepdfs(current_user):
        response = jsonify({"filegroups": get_all_filegroups()})
        return response

    @app.route("/api/get-file-groupdata/<path:filegroup>", methods=['POST'])
    @token_required
    def pdfimage(current_user, filegroup):
        fg = get_filegroup_data(filegroup)
        dec = ''
        if fg['group_predicted_class'] == 'N2+N4':
            dec = 'N2 ' + decisionclass['N2'] + '+' + 'N4 ' + decisionclass['N4']

        elif fg['group_predicted_class'] != '':
            dec = fg['group_predicted_class'] + ' ' + decisionclass[fg['group_predicted_class']]
        fls = []
        print(fg)
        for fl in fg['files']:
            ls = []
            fileinfo_id =fg['file_ids'][fg['files'].index(fl)]
            file_info_obj = model.FileInfo.query.filter(model.FileInfo.id==fileinfo_id).first()
            if file_info_obj != None:
                mdb = DbConf.mdb
                fs = GridFS(mdb)
                file_mongo_id = file_info_obj.file_data_id
                file_data_obj = mdb.get_collection('fileData').find({'_id': ObjectId(file_mongo_id)})
                #print("File Data Object==> ",dumps(file_data_obj))
                #get mongo object 
                for doc in file_data_obj:
                    print(doc['image_file'],type(doc['image_file']),str(doc['image_file']))
                    r = doc['image_file']
                    outputdata =fs.get(r).read()
                    
                    img_64_data = base64.b64encode(outputdata).decode()
                    # img.getvalue()
                    # print("=========   ",type(image_data))
                    # print("=========   ",image_data)
                    # print("=========   ",image_data.hex())
                    # response = make_response(image_data.hex())
                    # response.headers['Content-Type'] = 'image/jpeg'
                    # response.headers['Content-Disposition'] = 'attachment; filename=img.jpg'
                    # return response
            fls.append(
                {'filename': fl, 'uploadfilename': fl.rsplit('.', 1)[0] + '.jpg?' + str(time.time()),'img': img_64_data,
                 'Type': fg['filetypes'][fg['files'].index(fl)]})
            

        pred = {'decision': dec, 'auto': fg['Auto'], 'procedure': fg['Procedure_Type'], 'court': fg['Court'],
                'solicitor': fg['Solictor'], 'amount': fg['Amount'], 'days': fg['Time Frame'],
                'docDate': fg['Document date'], 'stampDate': fg["Stamp date"]}
        real = pred
        realList = ['decision(Actual)', 'Auto(Actual)', 'Procedure_Type(Actual)', 'Court(Actual)', 'Solictor(Actual)',
                    'Amount(Actual)', 'Time Frame(Actual)', 'Document date(Actual)', 'Stamp date(Actual)']
        f = True
        for r in realList:
            if not r in fg.keys():
                f = False
        if f:
            real = {'decision': fg['decision(Actual)'], 'auto': fg['Auto(Actual)'],
                    'procedure': fg['Procedure_Type(Actual)'], 'court': fg['Court(Actual)'],
                    'solicitor': fg['Solictor(Actual)'], 'amount': fg['Amount(Actual)'],
                    'days': fg['Time Frame(Actual)'], 'docDate': fg["Document date(Actual)"],
                    'stampDate': fg["Stamp date(Actual)"], 'keywords': str(fg['Keywords'])}

        response = jsonify({'files': fls,
                            "pred": pred,
                            "real": real,
                            })
        return response

    @app.route('/api/input-feedback/<path:filegroup>', methods=['GET', 'POST'])
    @token_required
    def receive_feedback(current_user, filegroup):
        js = request.get_json(silent=True)
        save_inputfeedback(js, filegroup)
        return "success"

    @app.route('/api/upload/<path:filename>')
    @token_required
    def upload_file(current_user, filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


        # @app.route('/api/input-feedback/<path:filegroup>', methods=['GET', 'POST'])
        # def receive_feedback(filegroup):
        #     print("here")
        #     data = request.get_json()
        #     x_auth = data['xauth']
        #     if check_token(x_auth):
        #         js = request.get_json(silent=True)
        #         save_inputfeedback(js, filegroup)
        #         return "success"
        #     resp = jsonify({"ErrorMessage": " 'User not logged In' "})
        #     resp.status_code = 204
        #     return resp

    @app.route('/')
    def index():
        return "Hello RACMO"

    CORS(app, resources=r'/api/*', headers='Content-Type')
    app.config['CORS_HEADERS'] = 'Content-Type'
    
    return app,model
