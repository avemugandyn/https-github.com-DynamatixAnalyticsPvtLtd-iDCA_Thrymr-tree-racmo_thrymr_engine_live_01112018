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
    
    @app.route("/api/save-user-details",methods=['POST'])
    def seveUserDetails():
        data = request.get_json()
        if data != None:
            try:
                username = data['username']
            except Exception as e:
                resp = jsonify({"MissingParameterError": " 'username' "})
                resp.status_code = 400
                return resp
            try:
                new_folder = data['new_folder_location']
            except Exception as e:
                resp = jsonify({"MissingParameterError": " 'New folder location for user' "})
                resp.status_code = 400
                return resp
            try:
                archive_folder = data['archive_folder_location']
            except Exception as e:
                resp = jsonify({"MissingParameterError": " 'Archive folder location for user' "})
                resp.status_code = 400
                return resp
        
            if not model.UserDetails.query.filter(model.UserDetails.name == username).first():
                new_user = model.UserDetails(name=username,
                                       new_folder=new_folder,
                                       archive_folder=archive_folder)
                model.db.session.add(new_user)
                model.db.session.commit()
                
                return jsonify("Successfully Save record ")
            else:
                resp = jsonify({"Error": " 'User already there !!' "})
                resp.status_code = 400
                return resp
        else:
            resp = jsonify({"Error": " 'Please send required data !!' "})
            resp.status_code = 400
            return resp
    
    @app.route("/api/run-analysis",methods=['POST'])
    def runAnalysis():
        data = request.get_json()
        if data != None:
            try:
                username = data['username']
            except Exception as e:
                resp = jsonify({"MissingParameterError": " 'username' "})
                resp.status_code = 400
                return resp
            user = model.UserDetails.query.filter(model.UserDetails.name == username).first()
            if user is not None:
                new_folder_location = user.new_folder
                archive_folder_location = user.archive_folder
                #set analysis function here and pass folder-path
                print(new_folder_location,archive_folder_location)
                ######
                try:
                    Document_Analysis.read_pdf_n_insert(new_folder_location,archive_folder_location,model)
                    return jsonify("Successfully Save record ")
                except Exception as e:
                    return jsonify(" Some thing went wrong ")
                ######
            else:
                resp = jsonify({"Error": " 'User not there !!' "})
                resp.status_code = 400
                return resp
        else:
            resp = jsonify({"Error": " 'Please send required data !!' "})
            resp.status_code = 400
            return resp
        
        
    
    @app.route('/')
    def index():
        return "Hello RACMO"
    
    CORS(app, resources=r'/api/*', headers='Content-Type')
    app.config['CORS_HEADERS'] = 'Content-Type'
    
    return app,model
