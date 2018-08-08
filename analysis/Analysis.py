import pandas as pd
import numpy as np
import pdfquery
import json
from shapely.geometry import box
from shapely.ops import cascaded_union
import pdftableextract as pte
from math import floor
from pymongo import MongoClient
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
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from configuration.configuration import ConfigClass,DbConf
import shutil
import datetime
import multiprocessing as mp
import datefinder

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
        print((len(tmp)))
      # assert tmp.apply(lambda x:x.file.split("_")[3] == x.group, axis = 1).all()
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
                print('rtf')
                newtext =textract.process(pf)
                newtext=str(newtext,'utf-8')
                # newtext=newtext
            else:
                newtext=Document_Analysis.get_pdf_text(pf)
            if(len(newtext.split())==0):
    #             print(''.join(newtext.split()))
                print("scan")

                newtext =textract.process(pf,method='tesseract')
    #             print(''.join(newtext.split()))
                # newtext=newtext
    #             print(''.join(newtext.split()))
            try:
                newtext=(''.join((c for c in unicodedata.normalize('NFD', newtext) if unicodedata.category(c) != 'Mn'))).lower()
            except Exception as e:
                print((str(e)+" binary ",pf[-3:].lower()))
                newtext=(''.join((c for c in unicodedata.normalize('NFD', newtext.decode("utf-8")) if unicodedata.category(c) != 'Mn'))).lower()
            # if pf[-3:].lower()=='rtf':
            #     print("rtf ----------->",newtext)
            rem=''
            paratlist=['MODO DE IMPUGNACION:'.lower(),'mode d\'impugnacio',
                       'recurso de repelacion','recurs de reposicio','recurso de reposicion','recurso de apelacion']

            for parat in paratlist:
                if (parat.lower() in newtext) :
                    rem=newtext.split(parat.lower())[-1]
            newtext=newtext.replace(rem,'')
        except Exception as e:
            newtext='Error:'+str(e)
        return newtext
    
    #takes file data frame and returns its table response(table json) and text response
    def parsefile(fdf,PDF_DIR,co):
        for i, r in fdf.iterrows():
                ticresponse=""
                textresponse=""
                try:
                    ticresponse=Document_Analysis.parse_ticket(join(PDF_DIR,r.filename))
                    # if 'Error:' in ticresponse:
                    #     print(i,ticresponse)
                except Exception as e:
                    ticresponse='Error:'+str(e)
                try:
                    if r.filename[-3:].lower()!='zip':
                        if r.filetype=='TICKET':
                                textresponse=ticresponse
                        else:
                            textresponse=Document_Analysis.parse_other(join(PDF_DIR,r.filename))
                            if 'Error:' in textresponse:
                                print(("text",i,textresponse))
                    else:
                        textresponse=""
                        with zipfile.ZipFile(join(PDF_DIR,r.filename)) as z:
                            for fileinzip in z.namelist():

                                if not os.path.isdir(fileinzip):
                                    # read the file
                                    zfdir=join(PDF_DIR, os.path.basename(fileinzip))
                                    with z.open(fileinzip) as fz,open(zfdir, 'wb') as zfp:
                                                shutil.copyfileobj(fz, zfp)
                                                text=parse_other(join(PDF_DIR,zfdir))
                                                if text[:5]!='Error':
                                                    textresponse+=text
                                                os.remove(zfdir)
                        # print(textresponse)
                except Exception as e:
                    textresponse='Error:'+str(e)
                if (r.filename[-3:].lower()=='rtf'):
                    print("___________________________________________________")
                    # print(i,textresponse)
                    
                fdf.loc[i,"table_response"] = ticresponse
                fdf.loc[i,"text_response"] = textresponse
        return fdf
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
    #         if row['fileclass']=='N16':
    #         if bias=='N1':
    #              print(bias,filename,keyword)
            for i, kr in kdf[(kdf['id'].isin(row['sub']))].iterrows():
                f=True
                for k in kr['keyword']:
                    if not (''.join(unidecode.unidecode(k).split()).lower() in text):
                        f=False
                if f:

                    bias1=Document_Analysis.get_predclass_normal(kdf,kr,j,text,fdf)
    #                 print(bias,bias1,filename,keyword,k)
                    bias=bias1
                    if kr['fileclass'] in list(fdf.loc[j,'keywords'].keys()):
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
                    else:
                        fdf.loc[j,'keywords'][kr['fileclass']]=list()
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
        return bias
    
    
      # A recursive function used for classification based on the hierarchy of the keywords and and its occurence in a particuar file through fuzzy search of distance 1
    def get_predclass_fuzzy(kdf,row,j,text,fdf):
        bias=row['bias']
        if len(row['sub'])>0:
            for i, kr in kdf[(kdf['id'].isin(row['sub']))].iterrows():
                f=True
                for k in kr['keyword']:
                    try:
                        match=find_near_matches(''.join(unidecode.unidecode(k).split()).lower(), text, max_l_dist=1)
                    except Exception as e:
                        f=False
                    if len(match)==0:
                        f=False
                if f:
                    bias1=Document_Analysis.get_predclass_normal(kdf,kr,j,text,fdf)
    #                 print(bias,bias1,filename,keyword,k)
                    bias=bias1
                    if kr['fileclass'] in list(fdf.loc[j,'keywords'].keys()):
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
                    else:
                        fdf.loc[j,'keywords'][kr['fileclass']]=list()
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
        return bias

 #File classification based on notification bibles   
    def get_classify_result(fdf,kdf,suspkdf,notification_corelation_dict):
        kdf["document_list"]=np.empty((len(kdf), 0)).tolist()
        fdf["keywords"]=fdf["filename"].apply(lambda x:{})
        fdf["pred_class"]=fdf["filename"].apply(lambda x:list())
        fdf["final_categ"]=fdf["filename"].apply(lambda x:list())
        fdf["remove_class"]=fdf["filename"].apply(lambda x:list())
        fdf["after_classfn"]=fdf["filename"].apply(lambda x:list())
        fdf["final_categ"]=fdf["filename"].apply(lambda x:list())
        fdf.loc[fdf['filetype']=='TICKET','text_response']=fdf['table_response']
        NX_filename_N1_N5=['ICO','S3A','S05','S02','S02','S5L','S04','S01','CNA','S5C','PCO','ASE','S1C']
        classlis=[]
        for j,row in fdf.iterrows():
            if row['filetype']=='TICKET':
                text=''.join(unidecode.unidecode((row['text_response'])).split()).lower()
            else:
                text=''.join(row['text_response'].split())
            paratlist=['MODO DE IMPUGNACION:'.lower(),'mode d\'impugnacio',
                       'recurso de repelacion','recurs de reposicio','recurso de reposicion','recurso de apelacion']
            rem=''
            for parat in paratlist:
                rem_word = ''.join(parat.split()).lower()
                if (rem_word in text) :
                    rem=text.split(rem_word)[-1]
    
            text=text.replace(rem,'')
    #         text=text.split("impugnacion")[0]
            for i,kdrow in kdf.iterrows():
    #         #       
                if text[:5]!='Error':
                    f=True
                    for k in kdrow['keyword']:
                        if not (''.join(unidecode.unidecode(k).split()).lower() in text):
                            f=False
                    if f and (kdrow['filetype']==row['filetype'] or(kdrow['filetype']=='NOTIFICATION' and row['filetype']=='OTHER') )and (kdrow['purpose']=='CLASSIFICATION'):
                        fdf.loc[j,'pred_class'].append(Document_Analysis.get_predclass_normal(kdf,kdrow,j,text,fdf))
                        if kdrow['fileclass'] in list(fdf.loc[j,'keywords'].keys()):
                            fdf.loc[j,'keywords'][kdrow['fileclass']].append(kdrow['keyword'])
                        else:
                            fdf.loc[j,'keywords'][kdrow['fileclass']]=list()
                            fdf.loc[j,'keywords'][kdrow['fileclass']].append(kdrow['keyword'])

            if not bool(fdf.loc[j,'keywords']):
                for i,kdrow in kdf.iterrows():
                    if text[:5]!='Error':
                        f=False
                        if(kdrow['filetype']==row['filetype'] or(kdrow['filetype']=='NOTIFICATION' and row['filetype']=='OTHER') )and (kdrow['purpose']=='CLASSIFICATION'):
                            f=True
                            for k in kdrow['keyword']:
                                match=[]
                                try:
                                    match=find_near_matches(''.join(unidecode.unidecode(k).split()).lower(), text, max_l_dist=1)
                                except Exception as e:
                                    f=False
                                if len(match)==0:
                                    f=False
                        if f: 
                            for special_kw in ['pagos', 'transferiran']:
                                if not (special_kw in text):
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
                        if not (''.join(unidecode.unidecode(sk).split()).lower() in ''.join(row['text_response'].split() )):
                            f=False
                    if f and (row['filetype']=='NOTIFICATION' or row['filetype']=='OTHER'):
    #                     if sr['remove_class'] in fdf.loc[j,'pred_class']:
                            if  not ((sr['remove_class']=='N2' or sr['remove_class']=='N12') and (''.join(("SE ALZA LA SUSPENSION DE LAS ACTUACIONES").split()).lower() in text)) :

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
                    if flnx in fdf.loc[j,'filename'].split('_')[1]:
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
#Can be ignored


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


    def extract_data_from_filegroups(fdf,kdf,path):
        Procedure_Type_Mapping = {'131': 'HIP','176': 'CON','186': 'CON','1EH': 'HIP','1MO': 'MON','1NJ': 'ETN','1TJ': 'ETJ','2AM': 'DCO',
                     '2CS': 'EJE','AJM': 'AUX','ASE': 'CON','CAC': 'CON','CNA': 'CON','CNC': 'DCO','CNO': 'CON','CNS': 'CON','COG': 'COG',
                     'CON': 'CON','DPR': 'DP','EJC': 'EJE','EJH': 'HIP','ENJ': 'ETN','ETJ': 'ETJ','FIC': 'DCO','I62': 'CON','ICO': 'CON',
                     'JCB': 'CAM','JCU': 'JC','JVB': 'VER','MCC': 'MC','MNC': 'MCU','MON': 'MON','NUL': 'ETNJ','ORD': 'ORD','PCA': 'CON',
                     'PCI': 'CON','PCO': 'CON','PLD': 'PLD','POE': 'ETJ','POH': 'PSO','POJ': 'PSO','PTC': 'PTC','PTG': 'VER','RCA': 'RCS',
                     'RPL': 'RAP','S02': 'CON','S03': 'CON','S04': 'CON','S05': 'CON','S1C': 'CON','S2A': 'CON','S4P': 'CON','S5C': 'CON',
                     'S5L': 'CON','SC2': 'CON','SC4': 'CON','SC5': 'CON','SC6': 'CON','TCD': 'TDO','TMH': 'TMD','V14': 'CNJ','VRB': 'VER',
                     'X39': 'CNJ','X53': 'DCO'}
        numspan={"1":"uno","2":"dos","3":"tres","4":"cuatro","5":"cinco","6":"seis","7":"siete","8":"ocho","9":"nueve",\
                 "10":"diez","11":"once","12":"doce","13":"trece","14":"catorce","15":"quince","16":"dieciseis",\
                 "17":"diecisiete","18":"dieciocho","19":"diecinueve","20":"veinte","21":"veintiuno","22":"veintidós",\
                 "23":"veintitrés","24":"veinticuatro","25":"veinticinco","26":"veintiséis","27":"veintisiete",\
                 "28": "veintiocho","29"  :"veintinueve","30" : "treinta"}
        numcat={"1":"un","2":"dos","3":"tres","4":"quatre","5":"cinc","6":"sis","7":"set","8":"vuit","9":"nou","10":"deu",\
                "11":"onze","12":"dotze","13":"tretze","14":"catorze","15":"quinze","16":"setze","17":"disset","18":"divuit",\
                "19":"dinou","20" : "vint"}
        Solicitor_keyword = ['ALCOCER ANTON, DOLORES [783]','GARCIA ABASCAL, SUSANA [721]','MALAGON LOYO, SILVIA [2058]']

        debtor_extraction = [ "demandado , demandado , demandado , demandado d/na.","demandado , demandado d/na.","demandado d/na.","demandado: d./dna.",
                              "demandado:","demandado.","ejecutado , ejecutado , ejecutado d/na.","ejecutado: d./dna.","ejecutado d/dna.","ejecutado d/na.",
                              "ejecutado:","solicitante d/na","seguidos contra","instado por","nombre y apellidos","contra da","frente a","contra: d/na.",
                              "contra d/na.:","contra:d./dna","contra:","contra don","nombre:","nombre completo:","deudor:",
                             "de: d/na.","titular:","concursado::","concursada:",
                              "ejecutado","part demandada/executada:","part demandada","part demandada:","parte demandada/ejecutada:","parte demandada",
                              "parte ejecutada","parte/s demandada/s:","parte recurrida:","procurador de los tribunales y de",
                              "escrito procurtadpr contrario, esta en el cuerpo del texto","de: d/na.","demandado","d/na","/ejecutada:","d./dna.","/ejecutada"]
        amount_extraction = ['en reclamacion de','restara per pagar','por importe de','la quanitad de','la quantitat de',
                             'por valor de',"la quantitat d'",'ingreso de','la reclamacion asciende a','de saldos',
                             'la cantidad retenida a','la suma de','las sumas reclamadas y que son las siguentes: principal:',
                             'las sumas reclamadas y que son las siguientes: principal:','por  importe de',
                             'por las cantidades de','por las siguientes cantidades','principal reclamado','la cuantia de',
                             'las sumas de','por un importe de','el importe de','cantidad de','pago por']

        extKeywords=kdf[kdf['purpose']=='EXTRACTION']
        extKeywords['decision_type']=extKeywords['decision_type'].apply(lambda x : x.split('-')[1])
        months={"enero": "January", "febrero":"February", "marzo": "March","abril": "April",
                "mayo":"May","junio":"June","julio":"July","agosto":"August","septiembre":"September","octubre":"October",
                "noviembre":"November", "diciembre":"December"}

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
        fgdf['Date_of_hearing']=''
        fgdf['Debtor']=''
        fgdf['list_of_possible_debtor']=fgdf['Debtor'].apply(lambda x : [])
        fgdf['Court']=''
        fgdf['Solictor']=''
        c=0
        for fi,fr in fgdf.iterrows():
            auto=""
            match=None
            match1=set()
            match2=set()
            for i,r in fdf[(fdf['filetype']!='TICKET')&(fdf['filename'].isin(fr['files']))].iterrows():
                s = r['text_response']
                if len(s.strip())>0:
                    s=unidecode.unidecode('\n'.join(list([_f for _f in s.split('\n') if _f])).lower())
                    debtor=''
                    list_of_possible_debtor = set()
                    for deb_ex in debtor_extraction:
                        if deb_ex in s:
                            for d in s.split(deb_ex)[1:]: 
                                debtor =  d.split('\n')[0]
                                if len(d.split('\n'))>1:
                                    debtor_2lines=d.split('\n')[0]+" "+d.split('\n')[1]
                                if len(debtor.split())==0:
                                    debtor = d.split('\n')[1]
                                    if len(d.split('\n'))>2:
                                        debtor_2lines=d.split('\n')[1]+" "+d.split('\n')[2]
                                debtor = debtor.strip()
                                debtor=Document_Analysis.debtor_filter(debtor,debtor_extraction)
                                debtor_2lines=Document_Analysis.debtor_filter(debtor_2lines,debtor_extraction)

                                list_of_possible_debtor=list_of_possible_debtor.union(Document_Analysis.debtor_split(debtor,' y '))
                                list_of_possible_debtor=list_of_possible_debtor.union(Document_Analysis.debtor_split(debtor,','))
                                list_of_possible_debtor=list_of_possible_debtor.union(Document_Analysis.debtor_split(debtor_2lines,' y '))
                                list_of_possible_debtor=list_of_possible_debtor.union(Document_Analysis.debtor_split(debtor_2lines,','))

                    if r['filetype']=='NOTIFICATION' and len(''.join(debtor.split())) >1:
                        fgdf.loc[fi,'Debtor']=debtor.upper().strip()
                    if len(''.join(fgdf.loc[fi,'Debtor'].split())) <=1 and len(''.join(debtor.split())) >1:
                        fgdf.loc[fi,'Debtor']=debtor.upper().strip()
                    fgdf.loc[fi,'list_of_possible_debtor']+=list_of_possible_debtor

                    ptype=""
                    if ptype!="":
                        if 'procedimiento' in s.lower():
                            match=re.search(r'(\d{1,20}/\d{4})',s.split('procedimiento')[1])
                        else:
                            s1=''.join(s.split())
                            match1=set(re.findall(r'(\d{1,20}/\d{4})',s1))
                            match2={'/'.join(x.split('/')[1:])for x in re.findall(r'(\d{1,2}/\d{1,2}/\d{4})',s1)}
                    if not match is None :
                            auto=match.group(0)
                    elif not (match1-match2) is None :
                            auto=list(match1-match2)[0] if len(list(match1-match2))>0 else ''

            if('N1'in fr['predicted_classes'])or('N3'in fr['predicted_classes'])or('N4'in fr['predicted_classes'])or('N11'in fr['predicted_classes']):
                for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']!='TICKET')].iterrows():
                    text=''.join(r['text_response'].split())
                    for k in amount_extraction:
                        if r['text_response'][:5]!='Error':
                            f=True
                            t=text
                            k1=''.join(unidecode.unidecode(k).split()).lower()
                            if  k1 in t:
                                t=t[t.index(k1)+len(k1):]
                            else:
                                f=False
                            k1 = ' '.join(k.split())
                            if f :
                                tex=' '.join(r['text_response'].split())
                                am= unidecode.unidecode(tex[tex.index(k1)+len(k1):]).split()[0]
                                if am!='en' or len(''.join(am.split()))>1:
                                    if any(i.isdigit() for i in am):
                                        try:
                                            am = am.split('euros')[0]
                                        except Exception as e:
                                            print((str(e)))
                                        try:
                                            am = am.replace("¬,","")
                                        except:
                                            am = unidecode.unidecode(am).replace("!","")
                                        am = am.replace('.-','')
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
                                        fgdf.loc[fi,'Amount']=am

            if(('N9'in fr['predicted_classes'])or('N10'in fr['predicted_classes'])):
                nls, nlc,nlm= [],[],[]
                for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']!='TICKET')].iterrows():
                    text=' '.join(unidecode.unidecode(r['text_response']).split())

                    if r['text_response'][:5]!='Error':
                        f=True
                        t=text
                        k1='dias'
                        if  k1 in t.lower():
                            nls+=  [t.lower().split()[i-1] for i, x in enumerate(t.lower().split()) if 'dias' in x ]
                        elif  'dies' in t.lower():
                            nlc+=  [t.lower().split()[i-1] for i, x in enumerate(t.lower().split()) if 'dies' in x ]
                        elif  'mes' in t.lower():
                            nlm+=  [t.lower().split()[i-1] for i, x in enumerate(t.lower().split()) if 'mes' in x ]
                        else:
                                f=False
                        if f :
                            mini=100
                            ls=[]
                            if len(nls)>0:
                                fgdf.at[fi,'Numlist']=nls
                                ls=nls
                                numbers=numspan
                            elif len(nlc)>0:
                                fgdf.at[fi,'Numlist']=nlc
                                ls=nlc
                                numbers=numcat
                            elif len(nlm)>0:
                                fgdf.at[fi,'Numlist']=nlm
                                ls=nlm
                                numbers=numcat
                            for num in ls:
                                if num in list(numbers.values()):
                                    #n=int([x for x in numbers.keys() if numbers[x]==num  ][0])
                                    for x in list(numbers.keys()):
                                        if numbers[x]==num and num == 'un':
                                            n=int(1*30)
                                        elif numbers[x]==num :
                                            n = int(x)
                                    if n < mini:
                                        mini=n
                                elif num in list(numbers.keys()):
                                    if int(num) < mini:
                                        mini=int(num)
                                else:
                                    print(num)
                            if min!=100:
                                fgdf.loc[fi,'Time Frame']=mini

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
                        table2=json.loads(js['table_2'])
                        c+=1
                        fgdf.loc[fi,"Document date"]=js["table_1"][[x for x in list(js["table_1"].keys()) if 'Fecha' in x][0]]
                        fgdf.loc[fi,"Stamp date"]=table2['1'][[x for x in list(table2['1'].keys()) if 'Fecha' in x][0]]

                        t_text = r['text_response']
                        try:
                            for soli_n in Solicitor_keyword:
                                if soli_n in t_text:
                                    fgdf.loc[fi,'Solictor']= soli_n
                                    break
                                else:
                                    fgdf.loc[fi,'Solictor']= ''
                            fgdf.loc[fi,'Court']=js["table_1"]['Remitente'][[x for x in list(js["table_1"]['Remitente'].keys()) if str(x)[:6].lower()=='organo'][0]]
                        except Exception as e:
                            c+=1
                        if fgdf.loc[fi,'Court'] =='' or fgdf.loc[fi,'Court'] == None:
                            try:
                                fgdf.loc[fi,'Court']=js["table_1"]['Destinatarios'][[x for x in list(js["table_1"]['Destinatarios'].keys()) if str(x)[:6].lower()=='organo'][0]]
                            except:
                                print("Court name not in Destinatarios")
                        auto=""
                        try:
                            s=json.loads(unidecode.unidecode(r['table_response']))['table_1']['Datos del mensaje']['Procedimiento destino']
                        except Exception as e:
                            s=str( json.loads(r['table_response'])['table_1']['Datos del mensaje'])

                        match1=set(re.findall(r'(\d{1,20}/\d{4})',s))
                        match2={'/'.join(x.split('/')[1:])for x in re.findall(r'(\d{1,2}/\d{1,2}/\d{4})',s)}
                        if not (match1-match2) is None :
                            auto=list(match1-match2)[0]

                    else:
                        
                        try:
                            try:
                                ts = unidecode.unidecode(textract.process(path+"/"+r['filename']))
                            except Exception as e:
                                ts = unidecode.unidecode(textract.process(path+"/"+r['filename']).decode('utf-8'))
                                print((str(e)))
                            if r['filename'] == '2009_0000375_CNO_20181020773555820180508145715_011.pdf':
                                fgdf.loc[fi,'Court']=ts.split('Organo')[1].split('\n')[3]
                            else:
                                fgdf.loc[fi,'Court']=ts.split('Organo')[1].split('\n')[1]
                            for soli_n in Solicitor_keyword:
                                if soli_n in ts:
                                    fgdf.loc[fi,'Solictor']= soli_n
                                    break
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

            if auto=='':
                sp=fgdf.loc[fi,'files'][0].split('_')
                fgdf.loc[fi,'Auto']=str(int(sp[1]))+'/'+str(int(sp[0]))
            else:
                fgdf.loc[fi,'Auto']= str(int(auto.split('/')[0]))+'/'+str(int(auto.split('/')[1]))

        print(c)
        return fgdf


    def read_pdf_n_insert(root_new,root_archive,model):
        PDF_DIR = root_new
        print(("hii",root_new))
        pdf_files = [f for f in listdir(PDF_DIR)\
                     if isfile(join(PDF_DIR,  f)) ]
        if len(pdf_files)>0:
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
            fdf = Document_Analysis.get_structured_files_dataframe(flgdf)
            fdf=fdf.rename(columns={"file":"filename","group":"filegroup","type":"filetype"})
            print("parsing")
            
            pool = mp.Pool(processes=4)
            ind=int()
            n=len(fdf)
            results = [pool.apply(Document_Analysis.parsefile, args=(fdf[int(x*len(fdf)/n):int((x+1)*len(fdf)/n)],PDF_DIR,x,)) for x in range(0,n)]
            # output = [p.get() for p in results]
            fdf=pd.concat(results)
            # fdf=Document_Analysis.parsefile(fdf)
            fdf,fgdf=Document_Analysis.update_filetype(fdf)
            ############
            #mongoupdate here
            ############
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
                       'N11' : {'N1','N2','N3','N7','N8','N9','N10','N11','N13','N14','N15','N15'},
                       'N12' : {'N12'},
                       'N13' : {'N1','N2','N3','N4','N9','N10','N11','N13'},
                       'N14' : {'N4','N11','N14'},
                       'N15' : {'N2','N3','N4','N9','N10','N11','N15'},
                       'N16' : {'N2','N3','N4','N9','N10','N11','N16'}
            } 
            kdf,suspkdf=Document_Analysis.keywordimport()
            print("classification")
            fdf=Document_Analysis.get_classify_result(fdf,kdf,suspkdf,notification_corelation_dict)
            print("extraction")
            fgdf=Document_Analysis.extract_data_from_filegroups(fdf,kdf,root_new)

            max_v = model.db.session.query(model.db.func.max(model.ProccessLog.batch_id)).scalar()
            if max_v==None:
                newmax=1
            else:
                newmax=max_v+1
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
                                               stamp_date_initial =rr['Stamp date'], 
                                               stamp_date = rr['Stamp date'],
                                               auto =rr['Auto'],
                                               auto_initial = rr['Auto'],
                                               amount_initial =rr['Amount'], 
                                               amount = rr['Amount'],
                                               date_of_hearing_initial =rr['Date_of_hearing'], 
                                               date_of_hearing =rr['Date_of_hearing'],
                                               debtor_initial =rr['Debtor'],
                                               possible_debtors = json.dumps(rr['list_of_possible_debtor']),
                                               debtor = rr['Debtor'],
                                               batch_id=newmax,
                                               creation_date = datetime.datetime.now()
                                    )
                model.db.session.add(kk)
                model.db.session.commit()
            writer = pd.ExcelWriter(PDF_DIR+'/../'+'Sample File_Details.xlsx', engine='openpyxl')
            fdf[['filename','filegroup','filetype','keywords','pred_class','final_categ']].to_excel(writer,'Sheet1')
            writer.save()
            for i , r in fdf.iterrows():
                k = model.FileClassificationResult(file_name =r['filename'],
                                             file_group =r['filegroup'],
                                             file_type=r['filetype'],
                                             predicted_classes=json.dumps(r['final_categ']),
                                             keyword=json.dumps(r['keywords']),
                                             batch_id=newmax,
                                             creation_date = datetime.datetime.now())
                model.db.session.add(k)
                model.db.session.commit()
                # shutil.copy(join( PDF_DIR,r.filename),join(root_archive,r.filename))
                # os.remove(join( PDF_DIR,r.filename))
            return True
        else:
            return False
        
