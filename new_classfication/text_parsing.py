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
#from configuration.configuration import ConfigClass,DbConf
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
notification_corelation_dict = { 'N1' : {'N1','N4','N7','N9', 'N11','N13'},
                                 'N2' : {'N2','N4','N7','N8','N11','N13','N15','N16'},
                                 'N3' : {'N3','N4','N7','N8','N11','N13','N15','N16',},
                                 'N4' : {'N1','N2','N3','N4','N7','N8','N9','N10','N13','N14','N15','N16'},
                                 'N5' : {'N5'},
                                 'N6' : {'N6'},
                                 'N7' : {'N1','N2','N3','N4','N7','N11'},
                                 'N8' : {'N2','N3','N4','N8','N9','N10','N11'},
                                 'N9' : {'N4','N8','N9','N10','N11','N13','N15','N16','N17'},
                                 'N10' : {'N4','N8','N9','N10','N11','N12','N13','N15','N16','N17'},
                                 'N11' : {'N1','N2','N3','N7','N8','N9','N10','N11','N13','N14','N15','N16'},
                                 'N12' : {'N10','N12'},
                                 'N13' : {'N1','N2','N3','N4','N9','N10','N11','N13','N17'},
                                 'N14' : {'N4','N11','N14'},
                                 'N15' : {'N2','N3','N4','N9','N10','N11','N15','N16'},
                                 'N16' : {'N2','N3','N4','N9','N10','N11','N15','N16'},
                                 'N17':{'N9','N10','N13','N17'}
                               }  

class Document_Analysis:
    
# SELECT max(batch_id) FROM file_classification;
    #def keywordimport():
        #engine = create_engine(ConfigClass.SQLALCHEMY_DATABASE_URI)
        
        #susp_df= pd.read_sql_query('SELECT k.id,k.file_class as fileclass,k.file_type as filetype,\
                                   #k.keyword,k.remove_class FROM suspend_keywords k',engine)
        #keyword_df= pd.read_sql_query('SELECT k.id,k.file_class as fileclass,k.file_type as filetype,k.purpose,\
                                   # k.decision_type,k.keyword,k.bias as bias,k.sub as sub FROM keywords k',engine)
        #keyword_df['sub']=keyword_df['sub'].apply(lambda x : json.loads(x) if x!=None else [])
        #keyword_df['keyword']=keyword_df['keyword'].apply(lambda x : json.loads(x))

        #return keyword_df, susp_df

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

  
    def read_pdf_n_insert(root_new):
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

            temp_df = pd.DataFrame()
            classi_fdf = pd.DataFrame()
            extract_fdf = pd.DataFrame()
            return fdf
        else:
            return temp_df

if __name__ == '__main__':
    #read_pdf_n_insert(root_new)
    fdf = Document_Analysis.read_pdf_n_insert("/home/racmo/upload_N17/")
    if len(fdf) >0:
       fdf.to_excel("/home/thrymr/552-python-Workspace/shashank_classfication/upload_N17_parse_22_10b.xlsx")
    
        