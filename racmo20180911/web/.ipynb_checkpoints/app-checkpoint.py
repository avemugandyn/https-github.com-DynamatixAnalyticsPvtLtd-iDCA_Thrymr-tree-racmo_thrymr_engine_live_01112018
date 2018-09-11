from flask import Flask, request, jsonify, make_response, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

import os
from os.path import join
import json
# local packages
from models.models import Models
from configuration.configuration import DbConf,ConfigClass
from analysis.Analysis import Document_Analysis

def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(ConfigClass)
    model = Models(app)
    
    @app.route("/api/run-analysis",methods=['GET'])
    def runAnalysis():
        new_folder_location = ConfigClass.NEW_FOLDER
        archive_folder_location = ConfigClass.ARCHIVE_FOLDER
        #set analysis function here and pass folder-path
        print(new_folder_location,archive_folder_location)
        
        try:
            if Document_Analysis.read_pdf_n_insert(new_folder_location,archive_folder_location,model):
                return jsonify("Successfully Save record ")
            else:
                return jsonify("No files in  the folder")
        except Exception as e:
            resp = jsonify({"Error:"+str(e) })
            resp.status_code = 400
            return resp
        
    @app.route('/')
    def index():
        max_v = model.db.session.query(model.db.func.max(model.FileGroup.batch_id)).scalar()
        print("max_v",max_v)
        users = model.FileGroup.query.filter(model.FileGroup.batch_id == max_v).first()
        print(users)
        
        return "Hello RACMO"
    
    CORS(app, resources=r'/api/*', headers='Content-Type')
    app.config['CORS_HEADERS'] = 'Content-Type'
    
    return app,model
