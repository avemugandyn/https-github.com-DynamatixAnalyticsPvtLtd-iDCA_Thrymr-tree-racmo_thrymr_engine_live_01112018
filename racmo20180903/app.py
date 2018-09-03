from flask import Flask, request, jsonify, make_response, send_from_directory
from werkzeug.utils import secure_filename
#from flask_cors import CORS, cross_origin

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
    
    #@app.route("/api/run-analysis",methods=['GET'])
    def runAnalysis():
        new_folder_location = ConfigClass.NEW_FOLDER
        archive_folder_location = ConfigClass.ARCHIVE_FOLDER
        #set analysis function here and pass folder-path
        print(new_folder_location,archive_folder_location)
        
        try:
            if Document_Analysis.read_pdf_n_insert(new_folder_location,archive_folder_location,model):
                return "Successfully Save record "
            else:
                return jsonify("No files in  the folder")
        except Exception as e:
            #resp = jsonify({"Error":str(e) })
            #resp.status_code = 400
            return "Error: "+str(e) 
        
    
    #CORS(app, resources=r'/api/*', headers='Content-Type')
    #app.config['CORS_HEADERS'] = 'Content-Type'
    runAnalysis()
    return app,model


if __name__ == '__main__':
    create_app()