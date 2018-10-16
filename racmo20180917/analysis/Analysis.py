import pandas as pd
import numpy as np
import pdfquery
import json
from shapely.geometry import box
from shapely.ops import cascaded_union
import pdftableextract as pte
from math import floor
import os
from os import listdir
from os.path import isfile, join, isdir
import time
import re
import pickle
import hashlib
import unidecode
from fuzzysearch import find_near_matches
import textract
import unicodedata
from pymongo import MongoClient
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from configuration.configuration import ConfigClass,DbConf
import shutil
import datetime
import multiprocessing as mp
import datefinder
import zipfile
import signal

PDF_DIR = None
temp_fdf = pd.DataFrame()
classi_fdf = pd.DataFrame()
extract_fdf = pd.DataFrame()
notification_corelation_dict = { 'N1' : {'N1','N4','N7','N11','N13'},
                                 'N2' : {'N2','N4','N7','N8','N11','N13','N15','N16'},
                                 'N3' : {'N3','N4','N7','N8','N11','N13','N15','N16'},
                                 'N4' : {'N1','N2','N3','N4','N7','N8','N9','N10','N13','N14','N15','N16'},
                                 'N5' : {'N5'},
                                 'N6' : {'N6'},
                                 'N7' : {'N1','N2','N3','N4','N7','N11'},
                                 'N8' : {'N2','N3','N4','N8','N9','N10','N11'},
                                 'N9' : {'N4','N8','N9','N10','N11','N13','N15','N16'},
                                 'N10' : {'N4','N8','N9','N10','N11','N12','N13','N15','N16'},
                                 'N11' : {'N1','N2','N3','N7','N8','N9','N10','N11','N13','N14','N15','N16'},
                                 'N12' : {'N12'},
                                 'N13' : {'N1','N2','N3','N4','N9','N10','N11','N13'},
                                 'N14' : {'N4','N11','N14'},
                                 'N15' : {'N2','N3','N4','N9','N10','N11','N15'},
                                 'N16' : {'N2','N3','N4','N9','N10','N11','N16'}
                               } 

class Document_Analysis:
    
# SELECT max(batch_id) FROM file_classification;
    def keywordimport():
        engine = create_engine(ConfigClass.SQLALCHEMY_DATABASE_URI)
        
        susp_df= pd.read_sql_query('SELECT k.id,k.file_class as fileclass,k.file_type as filetype,\
                                   k.keyword,k.remove_class FROM suspend_keywords k',engine)
        keyword_df= pd.read_sql_query('SELECT k.id,k.file_class as fileclass,k.file_type as filetype,k.purpose,\
                                    k.decision_type,k.keyword,k.bias as bias,k.sub as sub FROM keywords k',engine)
        keyword_df['sub']=keyword_df['sub'].apply(lambda x : json.loads(x) if x!=None else [])
        keyword_df['keyword']=keyword_df['keyword'].apply(lambda x : json.loads(x))

        return keyword_df, susp_df

#Getting number of pages in a pdf file    
    def get_pgnum(filename):
        pdf = pdfquery.PDFQuery(ConfigClass.UPLOAD_FOLDER + "/" + filename)
        pdf.load()
        pgn = len(pdf.tree.getroot().getchildren())
        return pgn

 #Takes a dataframe of filegroups which also contains the list of files as its argument and returns a dataframe files with columns, name,file group and its filetype(right now its either ticket or other as the file with smallest name in a filegroup is ticket)
    def get_structured_files_dataframe(df):
        tmp = pd.DataFrame(columns=["file","group"])
        j = 0
        for i, r in df.iterrows():
            for f in r.files:
                tmp.loc[j] = [f, r.group]
                j = j+1
        tmp["ext"] = tmp.apply(lambda x : x.file.split(".")[-1],axis =1)
        tmp["length"] = tmp.apply(lambda x : len(x.file),axis =1)
        TI, OI = [], []
        for case, indices in list(tmp.groupby("group").groups.items()):
            ticket_index = tmp.loc[indices].length.values.argmin()
            for i, idx in enumerate(indices):
                if i == ticket_index:
                    TI.append(idx)
                else:
                    OI.append(idx)
        tmp['type'] = "OTHER"
        tmp.loc[TI,"type"] = "TICKET"
        return tmp

    #takes a df and returns to json, the df is taken from the data extracted from the tables of pdfs using pdf tables extract
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
        for key, idx in list(df.groupby([0]).groups.items()):
            js[key] = {}
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
                js[key] = df.loc[idx][[1,2]].values.tolist()
            if len(js[key]) == 1 and "_value" in js[key]:
                js[key] = js[key]["_value"]
        return js

    #takes the path of the pdf and returns its text using pdf query the text extracted will be sorted accoring to its y cooridinates of its bounding boxes
    def get_pdf_text(path):
        try:
            pdf=pdfquery.PDFQuery(path)
            pdf.load()
            pdftext=""
            pgn=len(pdf.tree.getroot().getchildren())
            for i in range(0,pgn):
                root = pdf.tree.getroot().getchildren()[i]
                npg=[]
                for node in root.iter():
                    try:
            #             if node.tag == "LTTextLineHorizontal" or node.tag == "LTTextBox" or node.tag=="LTTextBoxHorizontal":
                        if node.text and float(node.get("y1"))>50 and float(node.get("x1"))>50:
                            npg.append(node)
            #                     pdftext=pdftext+"\n"
                    except Exception as e:
                        print((node.tag, e))
                npg=sorted(npg, key=lambda x: float(x.get("x0")))
                npg=sorted(npg, key=lambda x: round(float(x.get("y0"))),reverse=True)
                if len(npg)>0:
                    prev_y=round(float(npg[0].get("y0")))
                    for x in npg: 
                        esc="\n"
                        if round(float(x.get("y0")))==prev_y:
                            esc="|"
        #                         print(x.get("y0"),x.get("y1"),x.text)
                        prev_y=round(float(x.get("y0")))
                        if len(x.text)> 0:
                            pdftext+=esc+x.text
            return pdftext
        except Exception as e:
            return('Error:'+str(e))

     #takes rtf files path and returns its text    
    def get_rtf_text(path):
        text = os.popen('unrtf --text '+path).read()
        return text

    #takes the path of a pdf and extract table 1 and table 2 of tickets and returns its json 
    def parse_ticket(pdf_path):
        _JSON = {"filepath" : pdf_path}
        try:
            pdf = pdfquery.PDFQuery(pdf_path)
            pdf.load()
            root = pdf.tree.getroot().getchildren()[0]
            page_box = [float(x) for x in root.get("bbox")[1:-1].split(",")]
            tables, _ =\
            list(zip(
                *sorted(
                    [(p.bounds,p.area) for p in cascaded_union(
                        [box(*[float(x) for x in node.get("bbox")[1:-1].split(",")]) for node in root.iter() if node.tag == "LTRect"]
                    )],
                    key = lambda x : -x[1]
                )
            ))
            X = page_box[2]
            Y = page_box[3]
            xf = 11.69/X
            yf = 8.27/Y
            t1, t2 = tables

            table_1_bbox = ":".join(map(str,(t1[0]*xf - 0.1, (Y - t1[3])*yf - 0.1, t1[2]*xf + 0.1, (Y - t1[1])*yf + 0.1)))
            table_2_bbox = ":".join(map(str,(t2[0]*xf - 0.1, (Y - t2[3])*yf - 0.1, t2[2]*xf + 0.1, (Y - t2[1])*yf + 0.1)))

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
            _JSON["table_1"] = Document_Analysis.df_to_json(df1)
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
                return('Error:'+str(e))
        return json.dumps(_JSON, ensure_ascii=False)

    #takes a path of a file(pdf and rtf) and extract its texts and remove its accents of spansish characters
    def parse_other(pf):
        text=""
        try:
            if pf[-3:].lower()=='rtf':
                #print('rtf')
                newtext =textract.process(pf)
                newtext=str(newtext,'utf-8')
                # newtext=newtext
            else:
                newtext=Document_Analysis.get_pdf_text(pf)
            if(len(newtext.split())==0):
                #print("scan")
                newtext =textract.process(pf,method='tesseract').decode('utf-8')
            try:
                newtext=(''.join((c for c in unicodedata.normalize('NFD', newtext) if unicodedata.category(c) != 'Mn'))).lower()
            except Exception as e:
                print((str(e)+" binary ",pf[-3:].lower()))
                newtext=(''.join((c for c in unicodedata.normalize('NFD', newtext.decode("utf-8")) if unicodedata.category(c) != 'Mn'))).lower()
            rem=''
            paratlist=['MODO DE IMPUGNACION:'.lower(),'mode d\'impugnacio',
                       'recurso de repelacion','recurs de reposicio','recurso de reposicion','recurso de apelacion',
                        'INTERPONER RECURSO DIRECTO DE REVISION']

            for parat in paratlist:
                if (parat.lower() in newtext) :
                    rem=newtext.split(parat.lower())[-1]
            newtext=newtext.replace(rem,'')
        except Exception as e:
            newtext='Error:'+str(e)
        return newtext

    def unzip_add(fdf,PDF_DIR):
        ffdf = fdf.copy()
        #import zipfile
        try:
            for i, r in fdf.iterrows():
                zip_dict ={}
                if r.filename[-3:].lower()=='zip':
                    z_files = []
                    #print((r.filename.split('.')[0]))
                    with zipfile.ZipFile(join(PDF_DIR,r.filename)) as z:
                        for fileinzip in [x for x in z.namelist()]:
                            if not os.path.isdir(fileinzip):
                                try:
                                    #fileinzip = unidecode.unidecode(fileinzip)
                                    zfdir=join(PDF_DIR, os.path.basename(fileinzip))
                                except Exception as e:
                                    fileinzip = unidecode.unidecode(fileinzip)
                                    zfdir=join(PDF_DIR, os.path.basename(fileinzip))
                                try:
                                    with z.open(os.path.join(fileinzip)) as fz,open(zfdir, 'wb') as zfp:
                                        shutil.copyfileobj(fz, zfp)
                                        #os.remove(zfdir)
                                        if fileinzip.count('.') > 1:
                                            condition = '.' in fileinzip
                                            while(condition):
                                                ind = fileinzip.find('.')
                                                if ind != fileinzip.rfind('.'):
                                                    fileinzip = fileinzip[0:ind] + '_' + fileinzip[ind+1:]
                                                else:
                                                    condition = False

                                        f_name = r.filename[:-4]+'_'+str(''.join(fileinzip.split()))
                                        f_name = f_name.replace('/','_')
                                        if r.filename[:-4] in f_name:
                                            if not os.path.exists(join(PDF_DIR,f_name)) and not os.path.exists(join(PDF_DIR,unidecode.unidecode(f_name))):
                                                shutil.move(zfdir, join(PDF_DIR,unidecode.unidecode(f_name)))
                                                z_files.append({"filename":unidecode.unidecode(f_name),"filegroup":r['filegroup'],\
                                                    "filetype":"OTHER",'ext':fileinzip[-3:],"length":len(fileinzip)})
                                            else:
                                                os.remove(zfdir)
                                except Exception as e:
                                     print(e)
                    zip_dict[r['filegroup']]= z_files                                                                  
                if zip_dict !={}:
                    for k,val in list(zip_dict.items()):
                        for v in val:
                            ffdf = ffdf.append(pd.Series(v), ignore_index=True) 
        except Exception as e:
            print(str(e))           
        return ffdf         
       

    #takes file data frame and returns its table response(table json) and text response
    def parsefile(file_name):
        global PDF_DIR
        global temp_fdf
        def timeout(signum, frame):
           raise ValueError('time out,a very specific bad thing happened.')
        signal.signal(signal.SIGALRM, timeout)
        df = temp_fdf.loc[temp_fdf['filename']==file_name]
        zips_filegroup = {}
        for i, r in df.iterrows():
            ticresponse=""
            textresponse=""
            zip_dict = []
            try:
                signal.alarm(45)
                while 1:
                    ticresponse=Document_Analysis.parse_ticket(join(PDF_DIR,r.filename))
                    if len(ticresponse)>5:
                        signal.alarm(0)
                        break
            except Exception as e:
                ticresponse='Error:'+str(e)
            try:
                signal.alarm(45)
                if r.filename[-3:].lower()!='zip':
                    while 1:
                        if r.filetype=='TICKET':
                            textresponse=ticresponse
                        else:
                            textresponse=Document_Analysis.parse_other(join(PDF_DIR,r.filename))
                            if 'Error:' in textresponse:
                                print(("text",i,textresponse))
                        if len(ticresponse)>5:
                            signal.alarm(0)
                            break
            except Exception as e:
                print(e)
                textresponse='Error:'+str(e)
                if (r.filename[-3:].lower()=='rtf'):
                    print("___________________________________________________")
                    
            df.loc[i,"table_response"] = ticresponse
            df.loc[i,"text_response"] = textresponse
        return df

    # Once the table json is extracted we know which is the principal notification file and moves on to update it
    def update_filetype(fdf):
        fgs=fdf.groupby('filegroup')
        fgdf=pd.DataFrame(columns=['filegroup'])
        i=0
        fgdf['files']=np.empty((len(fgs.groups), 0)).tolist()
        fgdf['filetypes']=np.empty((len(fgs.groups), 0)).tolist()
        for k,v in list(fgs.groups.items()):
            fgdf.loc[i,'filegroup']=k
            files=[]
            pcs=[]
            filetypes=[]
            for ind in v:
                    files.append(fdf.loc[ind,'filename'])
                    filetypes.append(fdf.loc[ind,'filetype'])
            fgdf.loc[i,'filetypes']=filetypes
            fgdf.loc[i,'files']=files
            i+=1
        for i ,r in fdf[~(fdf['table_response'].str.contains('Error:'))&(fdf['filetype']=='TICKET')].iterrows():
            if r['table_response'][:5]!='Error':
                js=json.loads(r['table_response'])
                pf=r['filename'].split('.')[0]+'_'+''.join(js['table_1']['Documentos'][0][0].split()).split('(Principal)')[0]
                fgf=[''.join(x.split())for x in fgdf.loc[fgdf['filegroup']==r['filegroup'],'files'].values[0] ]
                if pf in fgf:
                    fl=fgdf.loc[fgdf['filegroup']==r['filegroup'],'files'].values[0][fgf.index(pf)]
                    if "caratula" in pf.lower():
                        fdf.loc[fdf['filename'].str.contains(fl),'filetype']="CARATULA"
                    else:
                        fdf.loc[fdf['filename'].str.contains(fl),'filetype']="NOTIFICATION"
                    
        return fdf,fgdf

    # A recursive function used for classification based on the hierarchy of the keywords and and its occurence in a particuar file
    def get_predclass_normal(kdf,row,j,text,fdf):
        bias=row['bias']
        if len(row['sub'])>0:
            for i, kr in kdf[(kdf['id'].isin(row['sub']))].iterrows():
                f = True
                for k in kr['keyword']:
                    kw = unidecode.unidecode(k)
                    if not (re.search(r'\b' + kw + r'\b', text)):
                         f=False
                    if not f:
                        if (''.join(kw.split()).lower() in ''.join(text.split()).lower()):
                            f=True
                    if f:
                        bias1=Document_Analysis.get_predclass_normal(kdf,kr,j,text,fdf)
                        bias=bias1
                    if kr['fileclass'] in list(fdf.loc[j,'keywords'].keys()):
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
                    else:
                        fdf.loc[j,'keywords'][kr['fileclass']]=list()
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
        return bias

    def get_predclass_fuzzy(kdf,row,j,text,fdf):
        bias = row['bias']
        if len(row['sub'])>0:
            for i, kr in kdf[(kdf['id'].isin(row['sub']))].iterrows():
                f=True
                for k in kr['keyword']:
                    try:
                        match=find_near_matches(unidecode.unidecode(k).lower(), text, max_l_dist=1)
                    except Exception as e:
                        f=False
                    if len(match)==0:
                        f=False
                if f:
                    bias1=Document_Analysis.get_predclass_fuzzy(kdf,kr,j,text,fdf)
                    bias=bias1
                    if kr['fileclass'] in list(fdf.loc[j,'keywords'].keys()):
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
                    else:
                        fdf.loc[j,'keywords'][kr['fileclass']]=list()
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
        return bias

    #def get_classify_result(fdf,kdf,suspkdf,notification_corelation_dict):
    def get_classify_result(file_name):
        kdf,suspkdf=Document_Analysis.keywordimport()
        global classi_fdf
        fdf = classi_fdf
        global notification_corelation_dict
        kdf["document_list"]=np.empty((len(kdf), 0)).tolist()
        fdf["keywords"]=fdf["filename"].apply(lambda x:{})
        fdf["pred_class"]=fdf["filename"].apply(lambda x:list())
        fdf["final_categ"]=fdf["filename"].apply(lambda x:list())
        fdf["remove_class"]=fdf["filename"].apply(lambda x:list())
        fdf["after_classfn"]=fdf["filename"].apply(lambda x:list())
      

        fdf.loc[fdf['filetype']=='TICKET','text_response']=fdf['table_response']
        NX_filename_N1_N5=['S05','S02','CNO','S5L','S05','S04','S01','CNA','S5C','PCO','ASE','S3A','ICO','S1C']

        classlis=[]
        
        caratula_keywords = ['NOTIFICACION TELEMATICA', 'DENTRO DEL ARCHIVO COMPRIMIDO']
    
        n2_prec_keywords = ['PARTE DISPOSITIVA', 'ANTECEDENTES DE HECHO', 'DILIGENCIA DE ORDENACIÓN']

        fdf  = fdf.loc[fdf['filename']==file_name]
        for j, row in fdf.iterrows():
            if row['filetype']!='TICKET':
            
                text = (row['text_response']).lower()

                text = re.sub(' +',' ', text)
                text = text.replace('\n', '')
                text = text.replace('\xaa', '')
                paratlist=['MODO DE IMPUGNACION:', 'mode d\'impugnacio', 'recurso de repelacion',
                           'recurs de reposicio', 'recurso de reposicion', 'recurso de apelacion', 
                           'INTERPONER RECURSO DIRECTO DE REVISION']
                rem=''
                for parat in paratlist:

                    rem_word = parat.lower()
                    if (rem_word in text):
                        rem=text.split(rem_word)[-1]
                    text=text.replace(rem,'')
            
                if all(word.lower() in text for word in caratula_keywords):
                    fdf.loc[j, 'filetype'] = 'CARATULA'
        
                for i,kdrow in kdf.iterrows():
                    if text[:5]!='Error':
                        f=True
                        for k in kdrow['keyword']:
                            kw = unidecode.unidecode(k)
                            if not (re.search(r'\b' + kw + r'\b', text)):
                                f=False

                        if f and (kdrow['filetype']==row['filetype'] or(kdrow['filetype']=='NOTIFICATION' and row['filetype']=='OTHER') )and (kdrow['purpose']=='CLASSIFICATION'):
                            fdf.loc[j,'pred_class'].append(Document_Analysis.get_predclass_normal(kdf,kdrow,j,text,fdf))
                            if kdrow['fileclass'] in list(fdf.loc[j,'keywords'].keys()):
                                fdf.loc[j,'keywords'][kdrow['fileclass']].append(kdrow['keyword'])
                            else:
                                fdf.loc[j,'keywords'][kdrow['fileclass']]=list()
                                fdf.loc[j,'keywords'][kdrow['fileclass']].append(kdrow['keyword'])

                if not bool(fdf.loc[j,'keywords']):
                    special_keywords = ['facilite','despachar','admitir','oigase']
                    for i,kdrow in kdf.iterrows():
                        if text[:5]!='Error':
                            f=False
                            if(kdrow['filetype']==row['filetype'] or(kdrow['filetype']=='NOTIFICATION' and row['filetype']=='OTHER') )and (kdrow['purpose']=='CLASSIFICATION'):
                                f=True
                                for k in kdrow['keyword']: 
                                    kw = unidecode.unidecode(k)
                                    if (len(kw) < 5) or any(sp in kw.lower() for sp in special_keywords ):
                                        if not (re.search(r'\b' + kw + r'\b', text)):
                                            f=False
                                    else:
                                        match=[]
                                        try:
                                            match=find_near_matches(kw.lower(), text, max_l_dist=1)
                                        except Exception as e:
                                            f=False
                                        if len(match)==0:
                                            f=False
                            if f:
                                fdf.loc[j,'pred_class'].append(Document_Analysis.get_predclass_fuzzy(kdf,kdrow,j,text,fdf))
                                if kdrow['fileclass'] in list(fdf.loc[j,'keywords'].keys()):
                                    fdf.loc[j,'keywords'][kdrow['fileclass']].append(kdrow['keyword'])
                                else:
                                    fdf.loc[j,'keywords'][kdrow['fileclass']]=list()
                                    fdf.loc[j,'keywords'][kdrow['fileclass']].append(kdrow['keyword'])
                fdf.set_value(j,'pred_class',list(set(fdf.loc[j,'pred_class'])))

                for si,sr in suspkdf.iterrows():
                    f=True
                    for sk in sr['keyword']:
                        if not (''.join(unidecode.unidecode(sk).split()).lower() in ''.join(text.split() )):
                            f=False
                    if f and (row['filetype']=='NOTIFICATION' or row['filetype']=='OTHER'):
                        if  not ((sr['remove_class']=='N2' or sr['remove_class']=='N12')and("SE ALZA LA SUSPENSION DE LAS ACTUACIONES".lower() in text)):
                            if 'NX-'+sr['remove_class'] in list(fdf.loc[j,'keywords'].keys()):
                                fdf.loc[j,'keywords']['NX-'+sr['remove_class']].append(sr['keyword'])
                            else:
                                fdf.loc[j,'keywords']['NX-'+sr['remove_class']]=list()
                                fdf.loc[j,'keywords']['NX-'+sr['remove_class']].append(sr['keyword'])
                            fdf.loc[j,'remove_class'].append(sr['remove_class'])
                f=True
                fdf.set_value(j,'remove_class',list(set(fdf.loc[j,'remove_class'])))
                if not 'N-ALL' in fdf.loc[j,'remove_class']:
                    for cl in list(set(fdf.loc[j,'pred_class'])-set(fdf.loc[j,'remove_class'])):
                        fdf.loc[j,'after_classfn'].append(cl)
                for cla in list(fdf.loc[j,'after_classfn'] ): 
                    s=set(notification_corelation_dict[cla])
                    if not(set(row['after_classfn'])<=(s) and set([cla])<=set(row['after_classfn'])):
                        f=False
                if f:
                    fdf.set_value(j,'final_categ',list(set(fdf.loc[j,'after_classfn'])))
                if 'N1' in fdf.loc[j,'final_categ'] or 'N5' in fdf.loc[j,'final_categ']:
                    fl=False
                    for flnx in NX_filename_N1_N5:
                        if flnx in fdf.loc[j,'filename'].split('_')[2]:
                            fl=True
                    if fl:
                        if 'N1' in fdf.loc[j,'final_categ']:
                            fdf.loc[j,'final_categ'].remove('N1')
                        elif 'N5' in fdf.loc[j,'final_categ']:
                            fdf.loc[j,'final_categ'].remove('N5')
       
        fgs=fdf.groupby('filegroup')
        for k,v in list(fgs.groups.items()):
            f=False
            otherclass="N16"
            for ind in v:
                if 'N16' in fdf.loc[ind,'final_categ']:
                    f=True
                if 'N8' in fdf.loc[ind,'final_categ']:
                    otherclass='N8'
                elif 'N15' in fdf.loc[ind,'final_categ']:
                    otherclass='N15'
            for ind in v:
                if f:
                    for n, i in enumerate(fdf.loc[ind,'final_categ']):
                        if i == 'N16':
                            fdf.loc[ind,'final_categ'][n] = otherclass
                        
        return fdf


    #Extraction relevant fields from diffeent files of filegroup based on extraction bible

    def debtor_filter(debtor,debtor_extraction):
        try:
            debtor = debtor.split('|')[1]
            debtor = debtor.strip()
        except Exception as e:
            debtor = debtor.strip()

        debtor = debtor.replace(':','')
        if  len(debtor)>4 and debtor[0]=='.':
            debtor=debtor[1:]
            debtor = debtor.strip()
        try:
            for deb_in in debtor_extraction:
                if deb_in in debtor:
                    debtor = debtor.replace(deb_in,'').strip()
        except Exception as e:
            debtor = debtor.strip()
        return debtor

    def debtor_split(debtor,split_string):
        list_of_possible_debtor=set()
        if len(debtor)>4:
            list_of_possible_debtor.add(debtor.strip())
        try:
            d_new = debtor.split(split_string)
            for dd in d_new:
                if len(dd)>4:
                    list_of_possible_debtor.add(dd.strip())
        except:
            pass
        return list_of_possible_debtor

    def string_amount_to_numeric(am_txt):
        f_a = 0
        sp_no_dict = {'cero':'0','uno':'1','tres':'3','cuatro':'4','cinco':'5','seis':'6','siete':'7','ocho':'8','nueve':'9','diez':'10',
                    'once':'11','doce':'12','trece':'13','catorce':'14','quince':'15','dieciseis':'16','diecisiete':'17','dieciocho':'18',
                    'diecinueve':'19','veinte':'20','veintiuno':'21','veintidos':'22','veintitres':'23','veinticuatro':'24','veinticinco':'25',
                    'veintiseis':'26','veintisiete':'27','veintiocho':'28','veintinueve':'29','treinta':'30','cuarenta':'40','cincuenta':'50',
                    'sesenta':'60','setenta':'70','ochenta':'80','noventa':'90','cien':'100','ciento':'100','doscientos':'200','trescientos':'300',
                    'cuatrocientos':'400','quinientos':'500','seiscientos':'600','setecientos':'700','ochocientos':'800',
                    'novecientos':'900','mil':'1000','dos mil':'2000','cuatro mil':'4000','diez mil':'10000' }
        if 'centimos' in am_txt:
            am = am_txt.split('centimos')[0]
            if len(am)<150:
                f_am1, f_am2,f_am = 0, 0, 0
                if 'euros' in am:
                    try:
                        am_lst = am.split('euros')
                        first_part = am_lst[0].split()
                        sec_part = am_lst[1].split()
                        if 'mil' in first_part:
                            ind = first_part.index('mil')
                            up_val ,low_val= 0, 0
                            for z in [sp_no_dict[y] for y in [v1 for x,v1 in enumerate(first_part) if x < ind] if y in list(sp_no_dict.keys())]:
                                up_val = up_val+float(z)
                            for z in [sp_no_dict[y] for y in [v1 for x,v1 in enumerate(first_part) if x > ind] if y in list(sp_no_dict.keys())]:
                                low_val = low_val+float(z)
                            f_am1 = (float(sp_no_dict[first_part[ind]])*float(up_val))+float(low_val)
                        else: 
                            for v in first_part:
                                if v in list(sp_no_dict.keys()):
                                    f_am1 = f_am1 + float(sp_no_dict[v])
                        for v1 in sec_part:
                            if v1 !='y':
                                if v1 in list(sp_no_dict.keys()):
                                    f_am2 = f_am2 + float(sp_no_dict[v1])
                    except Exception as e:
                        print(str(e))
                f_a = float(f_am1) + (float(f_am2)/100)
        elif 'euros' in am_txt :
            am_lst = am_txt.split('euros')[0]
            if len(am_lst) <150:
                am_lst = am_lst.split()
                try:
                    if 'mil' in am_lst:
                        ind = am_lst.index('mil')
                        up_val ,low_val= 0, 0
                        for z in [sp_no_dict[y] for y in [v1 for x,v1 in enumerate(am_lst) if x < ind] if y in list(sp_no_dict.keys())]:
                            up_val = up_val+float(z)
                        for z in [sp_no_dict[y] for y in [v1 for x,v1 in enumerate(am_lst) if x > ind] if y in list(sp_no_dict.keys())]:
                            low_val = low_val+float(z)
                        f_a = (float(sp_no_dict[am_lst[ind]])*float(up_val))+float(low_val)
                    else:
                        for v in am_lst:
                            if v !='y':
                                if v in list(sp_no_dict.keys()):
                                    f_a = f_a + float(sp_no_dict[v])
                except Exception as e:
                    print(str(e))
        return f_a
    
    
#    def extract_data_from_filegroups(fdf,path):
    def extract_data_from_filegroups(file_grp):
        global PDF_DIR
        global extract_fdf
        kdf,suspkdf = Document_Analysis.keywordimport()
        fdf = extract_fdf.loc[extract_fdf['filegroup']==file_grp]
        path = PDF_DIR
        Procedure_Type_Mapping = {'131': 'HIP','176': 'CON','186': 'CON','1EH': 'HIP','1MO': 'MON','1NJ': 'ETN','1TJ': 'ETJ','2AM': 'DCO',
                     '2CS': 'EJE','AJM': 'AUX','ASE': 'CON','CAC': 'CON','CNA': 'CON','CNC': 'DCO','CNO': 'CON','CNS': 'CON','COG': 'COG',
                     'CON': 'CON','DPR': 'DP','EJC': 'EJE','EJH': 'HIP','ENJ': 'ETN','ETJ': 'ETJ','FIC': 'OTH','I62': 'CON','ICO': 'CON',
                     'JCB': 'CAM','JCU': 'JC','JVB': 'VER','MCC': 'MC','MNC': 'MCU','MON': 'MON','NUL': 'ETNJ','ORD': 'ORD','PCA': 'CON',
                     'PCI': 'CON','PCO': 'CON','PLD': 'PLD','POE': 'ETJ','POH': 'PSO','POJ': 'PSO','PTC': 'PTC','PTG': 'VER','RCA': 'RCS',
                     'RPL': 'RAP','S02': 'CON','S03': 'CON','S04': 'CON','S05': 'CON','S1C': 'CON','S2A': 'CON','S4P': 'CON','S5C': 'CON',
                     'S5L': 'CON','SC2': 'CON','SC4': 'CON','SC5': 'CON','SC6': 'CON','TCD': 'TDO','TMH': 'TMD','V14': 'CNJ','VRB': 'VER',
                     'X39': 'CNJ','X53': 'DCO','S3A':'CON','PMC':'MC','PIE':'OTH','CUP':'OTH','181':'CON','CAB':'OTH','CUA':'JC','196':'CON',
                     'ITC':'OTH','SCA':'CON','X00':'DP','192':'CON'}

        time_frame_day = {'catorce': '14','catorze': '14','cinc': '5','cinco': '5','cuatro': '4','deu': '10','diecinueve': '19','dieciocho': '18',
                          'dieciseis': '16','diecisiete': '17','diez': '10','dinou': '19','disset': '17','divuit': '18','doce': '12','dos': '2',
                          'dotze': '12','nou': '9','nueve': '9','ocho': '8','once': '11','onze': '11','quatre': '4','quince': '15','quinze': '15',
                          'seis': '6','set': '7','setze': '16','siete': '7','sis': '6','trece': '13','treinta': '30','tres': '3','tretze': '13',
                          'un': '1','uno': '1','veinte': '20','veinticinco': '25','veinticuatro': '24','veintidos': '22','veintinueve': '29',
                          'veintiocho': '28','veintiseis': '26','veintisiete': '27','veintitres': '23','veintiuno': '21','vint': '20','vuit': '8'
                         }

        Solicitor_keyword = ['ALCOCER ANTON, DOLORES [783]','GARCIA ABASCAL, SUSANA [721]','MALAGON LOYO, SILVIA [2058]']

        debtor_extraction = [ "demandado , demandado , demandado , demandado d/na.","demandado , demandado d/na.","demandado d/na.","demandado: d./dna.",
                              "demandado:","demandado.","ejecutado , ejecutado , ejecutado d/na.","ejecutado: d./dna.","ejecutado d/dna.","ejecutado d/na.",
                              "ejecutado:","solicitante d/na","seguidos contra","instado por","nombre y apellidos","contra da","frente a","contra: d/na.",
                              "contra d/na.:","contra:d./dna","contra:","contra don","nombre:","nombre completo:","deudor:","de: d/na.","titular:",
                              "concursado::","concursada:","ejecutado","part demandada/executada:","part demandada","part demandada:","parte demandada/ejecutada:",
                              "parte demandada","parte ejecutada","parte/s demandada/s:","parte recurrida:","procurador de los tribunales y de",
                              "escrito procurtadpr contrario, esta en el cuerpo del texto","de: d/na.","demandado","d/na","/ejecutada:","d./dna.","/ejecutada"]

        amount_extraction = ['en reclamacion de','restara per pagar','por importe de','la quanitad de','la quantitat de',
                             'por valor de',"la quantitat d'",'ingreso de','la reclamacion asciende a','de saldos',
                             'la cantidad retenida a','la suma de','las sumas reclamadas y que son las siguentes: principal:',
                             'las sumas reclamadas y que son las siguientes: principal:','por  importe de',
                             'por las cantidades de','por las siguientes cantidades','principal reclamado','la cuantia de',
                             'las sumas de','por un importe de','el importe de','cantidad de','pago por','la cantidad consignada de','por importe total de',
                             'la cantidad ingresada de','cuenta del principal','en la cuenta de consignaciones la cantidad','la cantidad consignada',
                             'por su importe de','por las cantiades de','las cantidades consignadas a cuenta del principal','de consignaciones de este juzgado, esto es']

        months={"enero": "January", "febrero":"February", "marzo": "March","abril": "April","mayo":"May","junio":"June","julio":"July","agosto":"August",
                "septiembre":"September","octubre":"October","noviembre":"November", "diciembre":"December"}

        extKeywords=kdf[kdf['purpose']=='EXTRACTION']
        extKeywords['decision_type']=extKeywords['decision_type'].apply(lambda x : x.split('-')[1])


        fgdf=pd.DataFrame(columns=['filegroup'])
        ddf=fdf.copy()
        fgs=ddf.groupby('filegroup')
        i=0
        fgdf['filetypes']=np.empty((len(fgs.groups), 0)).tolist()
        fgdf['files']=np.empty((len(fgs.groups), 0)).tolist()
        fgdf['predicted_classes']=np.empty((len(fgs.groups), 0)).tolist()
        for k,v in list(fgs.groups.items()):
            pcs, files, filetypes= [], [], []
            for ind in v:
                files.append(ddf.loc[ind,'filename'])
                filetypes.append(ddf.loc[ind,'filetype'])
                pcs+=(ddf.loc[ind,'final_categ'])
            fgdf.loc[i,'filegroup']=k
            fgdf.loc[i,'predicted_classes']+=pcs
            fgdf.loc[i,'filetypes']=filetypes
            fgdf.loc[i,'files']=files
            i+=1
        c=0
        fgdf['Numlist']=[[] for _ in range(len(fgdf))]
        fgdf['Time Frame']=''
        fgdf['Amount']=''
        fgdf['Amount_list']=fgdf['Amount'].apply(lambda x : [])
        fgdf['Date_of_hearing']=''
        fgdf['Debtor']=''
        fgdf['list_of_possible_debtor']=fgdf['Debtor'].apply(lambda x : [])
        fgdf['Court']=''
        fgdf['Solictor']=''
        fgdf['Send_date']=''
        fgdf['Document date']=''
        fgdf['Auto']=''
        fgdf['Procedure_Type']=''

        c=0
        for fi,fr in fgdf.iterrows():
            auto=""
            match=None
            match1=set()
            match2=set()
            for i,r in fdf[(fdf['filetype']!='TICKET')&(fdf['filetype']!='CARATULA')&(fdf['filename'].isin(fr['files']))].iterrows():
                if r['text_response'][:5]!='Error':
                    s = r['text_response']
                    if len(s.strip())>0:
                        list_of_possible_debtor = set()
                        try:
                            s=unidecode.unidecode('\n'.join(list([_f for _f in s.split('\n') if _f])).lower())
                            debtor=''
                            #list_of_possible_debtor = set()
                            for deb_ex in debtor_extraction:
                                if deb_ex in s:
                                    for d in s.split(deb_ex)[1:]: 
                                        debtor =  d.split('\n')[0]
                                        if len(d.split('\n'))>1:
                                            debtor_2lines=d.split('\n')[0]+" "+d.split('\n')[1]+" "+[x for x in s.split(deb_ex)[0].split('\n')
                                                                                         if len(x)>3][-1].strip()
                                        if len(debtor.split())==0:
                                            debtor = d.split('\n')[1]
                                            if len(d.split('\n'))>2:
                                                debtor_2lines=d.split('\n')[1]+" "+d.split('\n')[2]
                                                debtor_2lines=d.split('\n')[1]+" "+d.split('\n')[2]+" "+[x for x in s.split(deb_ex)[0].split('\n')
                                                                                             if len(x)>3][-1].strip()
                                        debtor = debtor.strip()
                                        debtor=Document_Analysis.debtor_filter(debtor,debtor_extraction)
                                        debtor_2lines=Document_Analysis.debtor_filter(debtor_2lines,debtor_extraction)

                                        list_of_possible_debtor=list_of_possible_debtor.union(Document_Analysis.debtor_split(debtor,' y '))
                                        list_of_possible_debtor=list_of_possible_debtor.union(Document_Analysis.debtor_split(debtor,','))
                                        list_of_possible_debtor=list_of_possible_debtor.union(Document_Analysis.debtor_split(debtor_2lines,' y '))
                                        list_of_possible_debtor=list_of_possible_debtor.union(Document_Analysis.debtor_split(debtor_2lines,','))
                        except Exception as e:
                            print(str(e))
                        try:
                            if r['filetype']=='NOTIFICATION' and len(''.join(debtor.split())) >1:
                                fgdf.loc[fi,'Debtor']=debtor.upper().strip()
                            if len(''.join(fgdf.loc[fi,'Debtor'].split())) <=1 and len(''.join(debtor.split())) >1:
                                fgdf.loc[fi,'Debtor']=debtor.upper().strip()
                            fgdf.loc[fi,'list_of_possible_debtor']+=list_of_possible_debtor
           
                        except Exception as e:
                            print(str(e))


            if('N1'in fr['predicted_classes'])or('N3'in fr['predicted_classes'])or('N4'in fr['predicted_classes'])or('N11'in fr['predicted_classes']):
                amount_lst = set()
                for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']!='TICKET')&(fdf['filetype']!='CARATULA')].iterrows():
                    text=''.join(r['text_response'].split()).lower()
                    for k in amount_extraction:
                        if r['text_response'][:5]!='Error':
                            f=True
                            t=text 
                            k1=''.join(unidecode.unidecode(k).split()).lower()
                            if  k1 in t:
                                t=t[t.index(k1)+len(k1):]
                            else:
                                f=False
    #                         k1 = ' '.join(kr['keyword'][0].split())
                            k1 = ' '.join(k.split())
                            if f :
                                #tex=' '.join(r['text_response'].split())
                                tex_lst=' '.join(r['text_response'].split()).split(k1)[1:]
                                for tex in tex_lst:
                                    #am= unidecode.unidecode(tex[tex.index(k1)+len(k1):]).split()[0]
                                    try:
                                        am= unidecode.unidecode(tex).split()[0]
                                        if len(''.join(am.split()))==1:
                                            am= unidecode.unidecode(tex).split()[1]
                                         #am= unidecode.unidecode(tex[tex.index(k1)+len(k1):]).split()[1]

                                        if am!='en' and len(''.join(am.split()))>1:
                                            if any(i.isdigit() for i in am):
                                                am = am.replace('euros','')
                                                am = re.sub(r'[?|$||!]',r'', am )
                                                am = am.replace('EUR','')
                                                am = am.replace('.-','')
                                                am = am.replace('(','')
                                                am = am.replace(')','')
                                                try:
                                                    am = am.replace("¬,","")
                                                except:
                                                    am = unidecode.unidecode(am).replace("!","")
                                                condition = '..' in am
                                                while condition:
                                                    am = am.replace('..','')
                                                    condition = '..' in am

                                                if ('.' in am or ',' in am) and (am[-1] == ',' or am[-1] == '.') :
                                                    am = am[:-1]
                                                try:
                                                    am = am.replace("´",",")
                                                except:
                                                    am = unidecode.unidecode(am)
                                                am = am.replace("'",",")
                                                if am.count(',') >= 1:
                                                    condition = ',' in am
                                                    while(condition):
                                                        ind = am.find(',')
                                                        am = am[0:ind] + '.' + am[ind+1:]
                                                        condition = ',' in am

                                                if am.count('.') > 1:
                                                    condition = '.' in am
                                                    while(condition):
                                                        ind = am.find('.')
                                                        if ind != am.rfind('.'):
                                                            am = am[0:ind] + '' + am[ind+1:]
                                                        else:
                                                            condition = False
                                                fgdf.loc[fi,'Amount']=am
                                                amount_lst.add(am)
                                            else:
                                                am_txt = unidecode.unidecode(tex)
                                                #am_txt = unidecode.unidecode(tex[tex.index(k1)+len(k1):])
                                                f_a = Document_Analysis.string_amount_to_numeric(am_txt)
                                                if f_a != 0:
                                                    fgdf.loc[fi,'Amount']=f_a
                                                    amount_lst.add(f_a)
                                    except Exception as e:
                                        print(str(e))
                fgdf.loc[fi,'Amount_list']+=list(amount_lst)

            if(('N9'in fr['predicted_classes'])or('N10'in fr['predicted_classes']) and ('N1' not in fr['predicted_classes'])):
                nls, nlc,nlm= [],[],[]
                final_tv = []
                for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']!='TICKET')].iterrows():
                    if 'modo impugnacion' in r['text_response']:
                        text=' '.join(unidecode.unidecode((r['text_response']).split('modo impugnacion')[0]).split())
                    elif 'modo de impugnacion' in r['text_response']:
                        text=' '.join(unidecode.unidecode((r['text_response']).split('modo de impugnacion')[0]).split())
                    elif 'recurso de reposicion' in r['text_response']:
                        text=' '.join(unidecode.unidecode((r['text_response']).split('recurso de reposicion')[0]).split())
                    else:
                        text=' '.join(unidecode.unidecode(r['text_response']).split())
                    if r['text_response'][:5]!='Error':
                        f=True
                        t=text

                        k1='dias'
                        if  k1 in t.lower():
                            nls+=  [t.lower().split()[i-1] for i, x in enumerate(t.lower().split()) if 'dias' in x ]
                            nls = [tt for tt in nls if len(tt) <25 ]
                        elif  'dies' in t.lower():
                            nlc+=  [t.lower().split()[i-1] for i, x in enumerate(t.lower().split()) if 'dies' in x ]
                            nlc = [tt for tt in nlc if len(tt) <25 ]
                        elif  'mes' in t.lower():
                            nlm+=  [t.lower().split()[i-1] for i, x in enumerate(t.lower().split()) if 'mes' in x ]
                            nlm = [tt for tt in nlm if len(tt)<25 ]
                        else:
                            f=False
                        if f :
                            if len(nls)>0:
                                fgdf.at[fi,'Numlist']=nls
                                for num in nls:
                                    if num in list(time_frame_day.values()):
                                            try:
                                                final_tv.append(int(num))
                                            except Exception as e:
                                                print(e)
                                    elif num in list(time_frame_day.keys()):
                                        for x in list(time_frame_day.keys()):
                                            if x == num :
                                                try:
                                                    final_tv.append(int(time_frame_day[x]))
                                                except Exception as e:
                                                    print(e)
                            elif len(nlc)>0:
                                fgdf.at[fi,'Numlist']=nlc
                                for num in nlc:
                                    if num in list(time_frame_day.values()):
                                        try:
                                            final_tv.append(int(num))
                                        except Exception as e:
                                            print(e) 
                                    elif num in list(time_frame_day.keys()):
                                        for x in list(time_frame_day.keys()):
                                            if x == num :
                                                try:
                                                    final_tv.append(int(time_frame_day[x]))
                                                except Exception as e:
                                                    print(e)
                                    else:
                                        print(num)
                            elif len(nlm)>0:
                                fgdf.at[fi,'Numlist']=nlm
                                for num in nlm:
                                    if num in list(time_frame_day.values()):
                                        try:
                                            final_tv.append(int(num)*30)
                                        except Exception as e:
                                            print(e)
                                    elif num in list(time_frame_day.keys()):
                                        for x in list(time_frame_day.keys()):
                                            if x == num :                
                                                try:
                                                    final_tv.append(int(time_frame_day[x])*30)
                                                except Exception as e:
                                                    print(e)

                        if len(final_tv) >0:
                            fgdf.loc[fi,'Time Frame']=final_tv[0]

            for i,r in fdf[(fdf['filetype']=='TICKET')&(fdf['filename'].isin(fr['files']))].iterrows():
                proced=''
                x=r.filename.split('_')[2]
                if x in  list(Procedure_Type_Mapping.keys()):
                    proced= Procedure_Type_Mapping[x]
                    fgdf.loc[fi,'Procedure_Type']=proced
                if r.table_response[:5]!='Error':
                    try:
                        js=json.loads( ''.join((c for c in unicodedata.normalize('NFD', r.table_response) if unicodedata.category(c) != 'Mn')))
                    except Exception as e:
                        print((str(e)))
                        js=json.loads(r.table_response)
                        
                    c+=1
                    try:
                        fgdf.loc[fi,"Document date"]=js["table_1"][[x for x in list(js["table_1"].keys()) if 'Fecha' in x][0]]
                    except Exception as e:
                        print(str(e))
                    try:
                        fgdf.loc[fi,"Send_date"]=js["table_1"][[x for x in list(js["table_1"].keys()) if 'Fecha-hora env' in x][0]]
                    except Exception as e:
                        print(str(e))
                    t_text = r['text_response']
                    try:
                        for soli_n in Solicitor_keyword:
                            if soli_n in t_text:
                                fgdf.loc[fi,'Solictor']= soli_n
                                break
                            else:
                                fgdf.loc[fi,'Solictor']= ''
                        #fgdf.loc[fi,'Court']=unidecode.unidecode(js["table_1"]['Remitente'][[x for x in list(js["table_1"]['Remitente'].keys()) if str(x)[:6].lower()=='organo'][0]])
                        court =unidecode.unidecode(js["table_1"]['Remitente'][[x for x in list(js["table_1"]['Remitente'].keys()) if unidecode.unidecode(str(x)[:6]).lower()=='organo'][0]]) 
                        m = re.search(r"\[([0-9]+)\]", court)
                        court = court.split("[")[0]+"["+str(int(m.group(1)))+"]"
                        fgdf.loc[fi,'Court']=court
                    except Exception as e:
                        print(str(e))
                        c+=1
                    if fgdf.loc[fi,'Court'] =='' or fgdf.loc[fi,'Court'] == None:
                        try:
                            court = unidecode.unidecode(js["table_1"]['Destinatarios'][[x for x in list(js["table_1"]['Destinatarios'].keys()) if unidecode.unidecode(str(x)[:6]).lower()=='organo'][0]])    
                            m = re.search(r"\[([0-9]+)\]", court)
                            court = court.split("[")[0]+"["+str(int(m.group(1)))+"]"
                            fgdf.loc[fi,'Court']= court
                            #fgdf.loc[fi,'Court']=unidecode.unidecode(js["table_1"]['Destinatarios'][[x for x in list(js["table_1"]['Destinatarios'].keys()) if str(x)[:6].lower()=='organo'][0]])
                        except Exception as e:
                            print(str(e),"Court name not in Destinatarios")

                else:
                    if r.filename[-3:].lower()!='zip':
                        ts = ''
                        try:
                            ts = unidecode.unidecode(textract.process(path+"/"+r['filename']))
                        except Exception as e:
                            print(str(e))
                        if ts =='':
                            try:
                                ts = unidecode.unidecode(textract.process(path+"/"+r['filename']).decode('utf-8'))
                            except Exception as e:
                                print("Error-",str(e))
                        if len(ts) >1:
                            try:
                                if 'Organo' in ts:
                                    court = ''
                                    try:
                                        court = ts.split('Organo')[1].split('\n')[1]
                                        if court =='':
                                            court = ts.split('Organo')[1].split('\n')[2]
                                            ss = ts.split('Organo')[1].split('\n')[3]
                                            if '[' in ss and ']' in ss:
                                            #m = re.search(r"\[([0-9]+)\]", ss)
                                                court = court + ' '+ss
                                        else:
                                            ss = ts.split('Organo')[1].split('\n')[2]
                                            if '[' in ss and ']' in ss:
                                            #m = re.search(r"\[([0-9]+)\]", ss)
                                                court = court + ' '+ss

                                        if 'Tipo' in court:
                                            court = ts.split('Organo')[1].split('\n')[3]
                                            if court == '':
                                                court = ts.split('Organo')[1].split('\n')[4]

                                        if 'signature not verified' in court.lower():
                                             court = ts.split('Organo')[1].split('\n')[8]
                                             ss = ts.split('Organo')[1].split('\n')[9]
                                             if '[' in ss and ']' in ss:
                                                court = court +' '+ ss
                                    except Exception as e:
                                        print((str(e)))
                                    try:
                                        m = re.search(r"\[([0-9]+)\]", court)
                                        court = court.split("[")[0]+"["+str(int(m.group(1)))+"]"
                                    except Exception as e:
                                        court = court
                                    fgdf.loc[fi,'Court']= court
                #                    fgdf.loc[fi,'Court']= court
                                elif "rrgano" in ts:
                                    try:
                                        court = ts.split('rrgano')[1].split('\n')[1]
                                        if court =='':
                                            court = ts.split('rrgano')[1].split('\n')[2]
                                            ss = ts.split('rrgano')[1].split('\n')[3]
                                            if '[' in ss and ']' in ss:
                                            #m = re.search(r"\[([0-9]+)\]", ss)
                                                court = court + ' '+ss
                                        else:
                                            ss = ts.split('rrgano')[1].split('\n')[2]
                                            if '[' in ss and ']' in ss:
                                            #m = re.search(r"\[([0-9]+)\]", ss)
                                                court = court + ', '+ss

                                        if 'Tipo' in court:
                                            court = ts.split('rrgano')[1].split('\n')[3]
                                            if court == '':
                                                court = ts.split('rrgano')[1].split('\n')[4]

                                        if 'signature not verified' in court.lower():
                                            court = ts.split('rrgano')[1].split('\n')[8]
                                            ss = ts.split('rrgano')[1].split('\n')[9]
                                            if '[' in ss and ']' in ss:
                                                court = court +' '+ ss

                                    except Exception as e:
                                        print((str(e)))
                                    try:
                                        m = re.search(r"\[([0-9]+)\]", court)
                                        court = court.split("[")[0]+"["+str(int(m.group(1)))+"]"
                                    except Exception as e:
                                        court = court
                                    fgdf.loc[fi,'Court']= court

                                for soli_n in Solicitor_keyword:
                                    if soli_n in ts:
                                        fgdf.loc[fi,'Solictor']= soli_n
                                        break
                                tt = ''.join(ts.split())
                                if 'Fecha-horaenv' in tt:
                                    dd = re.search(r'\d{2}/\d{2}/\d{4}\d{2}:\d{2}', tt.split('Fecha-horaenv')[1]).group(0)
                                    fgdf.loc[fi,"Send_date"] = re.search(r'\d{2}/\d{2}/\d{4}', dd).group(0)+" "+re.search(r'\d{2}:\d{2}', dd).group(0)
                            except Exception as e:
                                print((str(e)))
            if( 'N7' in fr['predicted_classes']):
                for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']=='NOTIFICATION')].iterrows():
                    text=r['text_response']
                    text=' '.join(text.split())
                    text=text.replace('del','')
                    text=text.replace('de','')
                    text=text.replace('a las','')
                    text=text.replace('a les','')
                    text=text.replace(' a ',' ')
                    text=text.replace('horas',' ')
                    text = ' '.join(text.split())

                    if "el proximo" in text or "el dia" in text:
                        if "el proximo" in text:
                            text=text.split("el proximo")[1]
                        else:
                            text=text.split("el dia")[1]
                        for k,v in list(months.items()):
                            text=text.replace(k,v)
                        f=True
                        matches=datefinder.find_dates(text)
                        for match in matches:
                            fgdf.loc[fi,"Date_of_hearing"]=str(match)
                            break

            if auto =='':
                sp=fgdf.loc[fi,'files'][0].split('_')
                try:
                    fgdf.loc[fi,'Auto']=str(int(sp[1]))+'/'+str(int(sp[0]))
                except Exception as e:
                    print(e)

        #print(c)
        return fgdf


    def read_pdf_n_insert(root_new,root_archive,model):
        global PDF_DIR
        global temp_fdf
        global classi_fdf
        global extract_fdf
        file_nm_error = list()
        import warnings
        warnings.filterwarnings("ignore")
        t_time = time.time()
        PDF_DIR = root_new
        pdf_files= [f for f in listdir(PDF_DIR) if isfile(join(PDF_DIR,  f))]
        if len(pdf_files)>0:
            ls=list()
            for pdf_file in pdf_files:
                if len(pdf_file.split('_'))>=4 and len(pdf_file.split('_')[2])<=4:
                    ls.append(pdf_file.split('_'))
                else:
                    file_nm_error.append(pdf_file)
            
            df=pd.DataFrame(ls)
            df = df[pd.notnull(df[3])]
            fg=list(df.groupby(3))
            ls=[]
            i=0
            for k,gdf in  fg:
                fglist=[]
                elist=[]
                flist=list()
                for i, row in gdf.iterrows():
                    row=row.dropna()
                    fln='_'.join(list(row))
                    flist.append(fln)
                    elist.append(fln[-3:].lower())
                fgroup={'group':k,'files':flist,'length':len(flist),'min_filename':min(flist, key=len),'extensions':elist}
                ls.append(fgroup)
            flgdf=pd.DataFrame(ls)
            flgdf=flgdf.dropna(thresh=1,axis=1)
            cmpt_fdf = Document_Analysis.get_structured_files_dataframe(flgdf)
            cmpt_fdf= cmpt_fdf.rename(columns={"file":"filename","group":"filegroup","type":"filetype"})
            cmpt_fdf = Document_Analysis.unzip_add(cmpt_fdf,PDF_DIR)
            temp_fdf = cmpt_fdf
            list_of_files = cmpt_fdf['filename'].values.tolist()
            print("Parsing start")
            parsing_start_t = time.time()
            pool = mp.Pool(processes=mp.cpu_count())
            results = pool.map_async(Document_Analysis.parsefile,list_of_files)
            pool.close()
            pool.join()
            print("Parsing complete(time in minutes) = ",(float(time.time()-parsing_start_t)/60))
            output = results.get()
            final_fdf=pd.concat(output)
           
            fdf,fgdf=Document_Analysis.update_filetype(final_fdf)
            ############
            #mongoupdate here
            ############
            ## notification_corelation_dict
            ############
            #kdf,suspkdf=Document_Analysis.keywordimport()
            print("Classification start")
            classifi_start_time = time.time()
            classi_fdf= fdf
            classi_files = cmpt_fdf['filename'].values.tolist()
            classification_pool = mp.Pool(processes=mp.cpu_count())
            classi_results = classification_pool.map_async(Document_Analysis.get_classify_result,classi_files)
            #fdf=pd.concat(cal_results)
            classification_pool.close()
            classification_pool.join()

            #fdf=Document_Analysis.get_classify_result(fdf,kdf,suspkdf,notification_corelation_dict)
            classi_output = classi_results.get()
            fdf= pd.concat(classi_output)
            print("Classification taking minutes = ",(float(time.time() - classifi_start_time)/60))
            print("Extraction start")
            extract_fdf = fdf
            ex_start_time = time.time()

            ext_filegroup_lst = extract_fdf['filegroup'].unique().tolist()
            extraction_pool = mp.Pool(processes=mp.cpu_count())
            ext_results = extraction_pool.map_async(Document_Analysis.extract_data_from_filegroups,ext_filegroup_lst)
            extraction_pool.close()
            extraction_pool.join()

            ext_output = ext_results.get()
            fgdf= pd.concat(ext_output)
            fgdf = fgdf.reset_index()
            #fgdf=Document_Analysis.extract_data_from_filegroups(fdf,root_new)
            if len(fgdf) != 0:
                max_v = model.db.session.query(model.db.func.max(model.ProccessLog.batch_id)).scalar()
                if max_v==None:
                    newmax=1
                else:
                    newmax = max_v+1
                print(("Batch_id= ",newmax))
                proccess_log = model.ProccessLog( batch_id=newmax,creation_date = datetime.datetime.now(),
                                                 process_date=datetime.datetime.now())
                model.db.session.add(proccess_log)
                model.db.session.commit()
                    # fgdf=fgdf.fillna('')
                fgdf=fgdf.applymap(lambda x: x if x==x else '')
                fgdf=fgdf.applymap(lambda x: None if x=='' else x)
                mdb = DbConf.mdb
                fileData = DbConf.fileData
                for i,mr in fdf.iterrows():
                    data = { "filename":mr['filename'],
                             "text_response": mr['text_response'],
                             "table_response": mr['table_response'],
                            }
                    fileData.insert_one(data)
                for i , rr in fgdf.iterrows():
                    timeframe=None
                    try:
                        if rr['Time Frame']==rr['Time Frame']:
                                timeframe=rr['Time Frame']
                        kk = model.FileGroup(file_group = rr['filegroup'],
                                                   court = rr['Court'],
                                                   court_initial = rr['Court'],
                                                   solicitor = rr['Solictor'],
                                                   solicitor_initial = rr['Solictor'],
                                                   procedure_type =rr['Procedure_Type'],
                                                   procedure_type_initial = rr['Procedure_Type'],
                                                   time_frame = timeframe,
                                                   document_date_initial =rr['Document date'],
                                                   document_date = rr['Document date'],
                                                   stamp_date_initial =rr['Send_date'], 
                                                   stamp_date = rr['Send_date'],
                                                   auto =rr['Auto'],
                                                   auto_initial = rr['Auto'],
                                                   amount_initial =rr['Amount'], 
                                                   amount = rr['Amount'],
                                                   date_of_hearing_initial =rr['Date_of_hearing'], 
                                                   date_of_hearing =rr['Date_of_hearing'],
                                                   debtor_initial =rr['Debtor'],
                                                   possible_debtors = json.dumps(rr['list_of_possible_debtor']),
                                                   possible_amount = json.dumps(rr['Amount_list']),
                                                   debtor = rr['Debtor'],
                                                   batch_id=newmax,
                                                   creation_date = datetime.datetime.now())
                        model.db.session.add(kk)
                        model.db.session.commit()
                    except Exception as e:
                        print("fgdf error-->",e)
                               
                for i , r in fdf.iterrows():
                    try:
                        if "error:time out,a very specific bad thing happened." in str(r['text_response']).lower():
                            k = model.FileClassificationResult(file_name =r['filename'],
                                                 file_group =r['filegroup'],
                                                 file_type=r['filetype'],
                                                 predicted_classes=json.dumps(r['final_categ']),
                                                 keyword=json.dumps(r['keywords']),
                                                 batch_id=newmax,
                                                 engine_comments = str(r['text_response']),
                                                 creation_date = datetime.datetime.now())
                        else:
                            k = model.FileClassificationResult(file_name =r['filename'],
                                                 file_group =r['filegroup'],
                                                 file_type=r['filetype'],
                                                 predicted_classes=json.dumps(r['final_categ']),
                                                 keyword=json.dumps(r['keywords']),
                                                 batch_id=newmax,
                                                 engine_comments ='',
                                                 creation_date = datetime.datetime.now())
                        model.db.session.add(k)
                        model.db.session.commit()
                        shutil.copy(join( PDF_DIR,r.filename),join(root_archive,r.filename))
                        os.remove(join( PDF_DIR,r.filename))
                    except Exception as e:
                        print(e)
                if len(file_nm_error)>0:
                    for f_nm in file_nm_error:
                        try:
                            kk = model.FileClassificationResult(file_name =f_nm,
                                                 file_group ='',
                                                 file_type=f_nm[-3:],
                                                 predicted_classes='',
                                                 keyword='',
                                                 batch_id=newmax,
                                                 engine_comments ='File name error,please verify manualy',
                                                 creation_date = datetime.datetime.now())
                            model.db.session.add(kk)
                            model.db.session.commit()
                            shutil.copy(join( PDF_DIR,f_nm),join(root_archive,f_nm))
                            os.remove(join( PDF_DIR,f_nm))
                        except Exception as e:
                            print(e)

            print("Extraction process time ---",(float(time.time() - ex_start_time)/60))
            print("Total time in minutes ---",(float(time.time() - t_time)/60))
            model.db.session.close()
            DbConf.client.close() # for close mongoDb connection
            temp_df = pd.DataFrame()
            classi_fdf = pd.DataFrame()
            extract_fdf = pd.DataFrame()
            return True
        else:
            return False
        