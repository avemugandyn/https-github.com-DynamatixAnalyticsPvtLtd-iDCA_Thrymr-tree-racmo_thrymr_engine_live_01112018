import pandas as pd
import numpy as np
import pdfquery
import json
from shapely.geometry import box
from shapely.ops import cascaded_union
import pdftableextract as pte
from math import floor
import pickle
import re
from pymongo import MongoClient
import os
from os import listdir
from os.path import isfile, join, isdir
import textract
import unidecode
from fuzzysearch import find_near_matches

import hashlib
import ast
import json
import unicodedata
import time
import PIL
from gridfs import GridFS
from PIL import Image
from shutil import copyfile

from configuration.configuration import ConfigClass,PyDbLite,DbConf

pyDbLite_db = PyDbLite().pyDbLite_db
fdf=pyDbLite_db.records[1]['file_filegroup']['file']
fgdf=pyDbLite_db.records[1]['file_filegroup']['filegroup_result']
PvRdf=pyDbLite_db.records[1]["result_df"]
# for i,r in pyDbLite_db.records[1]['file_filegroup']['file'].iterrows():
#     print(r['table_response'],r['filetype'])
def get_pgnum(filename):
    pdf = pdfquery.PDFQuery(ConfigClass.UPLOAD_FOLDER + "/" + filename)
    pdf.load()
    pgn = len(pdf.tree.getroot().getchildren())
    return pgn

def get_structured_files_dataframe(df):
    tmp = pd.DataFrame(columns=["file","group"])
    j = 0
    for i, r in df.iterrows():
        for f in r.files:
            tmp.loc[j] = [f, r.group]
            j = j+1
#     assert tmp.apply(lambda x:x.file.split("_")[3] == x.group, axis = 1).all()
    tmp["ext"] = tmp.apply(lambda x : x.file.split(".")[-1],axis =1)
    tmp["length"] = tmp.apply(lambda x : len(x.file),axis =1)
    TI, OI = [], []
    for case, indices in tmp.groupby("group").groups.items():
        ticket_index = tmp.loc[indices].length.values.argmin()
        for i, idx in enumerate(indices):
            if i == ticket_index:
                TI.append(idx)
            else:
                OI.append(idx)

    tmp['type'] = "OTHER"
    tmp.loc[TI,"type"] = "TICKET"
    return tmp

def df_to_json(df):
    js = {}
    vals = df[0].values
    ls = []
    for i, v in enumerate(vals):
        if not v:
            ls.append(ls[-1])
        else:
            ls.append(v)
    df[0] = ls
    for key, idx in df.groupby([0]).groups.items():
        js[key] = {}
        print(key,type("key"))
        for val in df.loc[idx][[1,2]].values:
            if val[0] == "" and val[1] == "":
                js[key]["_value"] = ""
            elif val[0] == "" and val[1] != "":
                js[key]["_value"] = val[1]
            elif val[0] != "" and val[1] == "":
                js[key]["_value"] = val[0]
            elif val[0] != "" and val[1] != "":
                js[key][val[0]] = val[1]
        if(key=="Documentos"):
            js[key] = [str(x) for x in df.loc[idx][[1,2]].values.tolist()]
        if len(js[key]) == 1 and "_value" in js[key]:
            js[key] = js[key]["_value"]
    return js

def parse_ticket(pdf_path):
    _JSON = {"filepath" : pdf_path}
    try:
            pdf = pdfquery.PDFQuery(pdf_path)
            pdf.load()
            root = pdf.tree.getroot().getchildren()[0]
            page_box = [float(x) for x in root.get("bbox")[1:-1].split(",")]
            tables, _ =\
            zip(
                *sorted(
                    [(p.bounds,p.area) for p in cascaded_union(
                        [box(*[float(x) for x in node.get("bbox")[1:-1].split(",")]) for node in root.iter() if node.tag == "LTRect"]
                    )],
                    key = lambda x : -x[1]
                )
            )
            X = page_box[2]
            Y = page_box[3]
            xf = 11.69/X
            yf = 8.27/Y
            t1, t2 = tables

            table_1_bbox = ":".join(map(str,(t1[0]*xf - 0.1, (Y - t1[3])*yf - 0.1, t1[2]*xf + 0.1, (Y - t1[1])*yf + 0.1)))
            table_2_bbox = ":".join(map(str,(t2[0]*xf - 0.1, (Y - t2[3])*yf - 0.1, t2[2]*xf + 0.1, (Y - t2[1])*yf + 0.1)))
            pte.process_page(
                        pdf_path,
                        "1",
                        crop = table_1_bbox,
                        pad=20
                    )

            df1 =\
            pd.DataFrame(
                pte.table_to_list(
                    pte.process_page(
                        pdf_path,
                        "1",
                        crop = table_1_bbox,
                        pad=20
                    ),
                    "1"
                )[1]
            )
            _JSON["table_1"] = df_to_json(df1)
            df2 = \
            pd.DataFrame(               
                pte.table_to_list(
                    pte.process_page(
                        pdf_path,
                        "1",
                        crop = table_2_bbox,
                        pad=20
                    ),
                    "1"
                )[1]
            )
            df2.columns = df2.iloc[0]
            df2 = df2.reindex(df2.index.drop(0))
            _JSON["table_2"] = df2.to_json(orient='index')
    except Exception as e:
        print(str(e))
        return('Error:'+str(e))

    return json.dumps(_JSON, ensure_ascii=False)
def read_pdf_n_insert(pdf_dir_root):
    PDF_DIR = pdf_dir_root
    pdf_files = [f for f in listdir(PDF_DIR)\
                 if isfile(join(PDF_DIR,  f)) ]
    ls=list()
    for pdf_file in pdf_files:
                ls.append(pdf_file.split('_'))
    df=pd.DataFrame(ls)
    df = df[pd.notnull(df[3])]
    fg=list(df.groupby(3))
    ls=[]
    i=0
    for k,gdf in  fg:
        fglist=[]
        elist=[]
        flist=[]
        for i, row in gdf.iterrows():
            row=row.dropna()
            fln='_'.join(list(row))
            flist.append(fln)
            elist.append(fln[-3:].lower())
        fgroup={'group':k,'files':flist,'length':len(flist),'min_filename':min(flist, key=len),'extensions':elist}
        ls.append(fgroup)
    flgdf=pd.DataFrame(ls)
    flgdf=flgdf.dropna(thresh=1,axis=1)
    fdf = get_structured_files_dataframe(flgdf)
    fdf=fdf.rename(columns={"file":"filename","group":"filegroup","type":"filetype"})

    for i, r in fdf[1:].iterrows():
        ticresponse=""
        textresponse=""
        try:
            ticresponse=parse_ticket(join(PDF_DIR,r.filename))
        except Exception as e:
            ticresponse='Error:'+str(e)
        print(ticresponse)
pyDbLite_db = PyDbLite().pyDbLite_db

read_pdf_n_insert("/home/thrymr/Racmo/processed/Test")
