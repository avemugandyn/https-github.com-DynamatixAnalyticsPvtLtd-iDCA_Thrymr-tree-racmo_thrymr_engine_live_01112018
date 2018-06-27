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
import ast
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from configuration.configuration import ConfigClass,DbConf
import shutil
import datetime

class Document_Analysis:
    def loadSession(engine):
        Base = declarative_base(engine)
        metadata = Base.metadata
        Session = sessionmaker(bind=engine)
        session = Session()
        return session

# SELECT max(batch_id) FROM file_classification;
    def keywordimport():
        engine = create_engine(ConfigClass.SQLALCHEMY_DATABASE_URI)
        session = Document_Analysis.loadSession(engine)
        susp_df= pd.read_sql_query('SELECT k.id,k.file_class as fileclass,k.file_type as filetype,\
                                   k.keyword,k.remove_class FROM suspend_keywords k',engine)
        keyword_df= pd.read_sql_query('SELECT k.id,k.file_class as fileclass,k.file_type as filetype,k.purpose,\
                                    k.decision_type,k.keyword,k.bias as bias,k.sub as sub FROM keywords k',engine)
        keyword_df['sub']=keyword_df['sub'].apply(lambda x : json.loads(x) if x!=None else [])
        keyword_df['keyword']=keyword_df['keyword'].apply(lambda x : json.loads(x))

        return keyword_df,susp_df

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
      # assert tmp.apply(lambda x:x.file.split("_")[3] == x.group, axis = 1).all()
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

    def get_pdf_text_2(path):
        try:
            pdf=pdfquery.PDFQuery(path)
            pdf.load()
            pdftext=""
            pgn=len(pdf.tree.getroot().getchildren())
            for i in range(0,pgn):
                root = pdf.tree.getroot().getchildren()[i]
                for node in root.iter():
                    try:
        #                 if node.tag == "LTTextLineHorizontal" or node.tag == "LTTextBox":
                        if node.text:
                            pdftext=pdftext+" "+node.text
        #                     pdftext=pdftext+"\n"
                    except Exception as e:
                        print(node.tag, e)
            return pdftext
        except Exception as e:
            return('Error:'+str(e))

    def get_rtf_text(path):
        text = os.popen('unrtf --text '+path).read()
        return text

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

    def parse_other(pf):
        text=""
        try:
            newtext =str(textract.process(pf),'utf-8')
            if(len(newtext.split())==0):
                newtext =str(textract.process(pf,method='tesseract'),'utf-8')
            newtext=(''.join((c for c in unicodedata.normalize('NFD', newtext) if unicodedata.category(c) != 'Mn'))).lower()
            rem=''
            paratlist=['MODO DE IMPUGNACION:'.lower(),'mode d\'impugnacio',
                       'recurso de repelacion','recurs de reposicio','recurso de reposicion','recurso de apelacion']
            for parat in paratlist:
                if (parat.lower() in newtext) :
                    rem=re.split(r'\s(?=(?:y firmo|y firma))',newtext.split(parat.lower())[-1],1)[0]
            newtext=newtext.replace(rem,'')
        except Exception as e:
            newtext='Error:'+str(e)
        return newtext

    def update_filetype(fdf):
        fgs=fdf.groupby('filegroup')
        fgdf=pd.DataFrame(columns=['filegroup'])
        i=0
        fgdf['files']=np.empty((len(fgs.groups), 0)).tolist()
        fgdf['filetypes']=np.empty((len(fgs.groups), 0)).tolist()
        for k,v in fgs.groups.items():
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
                    fdf.loc[fdf['filename'].str.contains(fl),'filetype']="NOTIFICATION"
        return fdf,fgdf

    def get_predclass_normal(kdf,row,j,text,fdf):
        bias=row['bias']
        print(row)
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
                    if kr['fileclass'] in fdf.loc[j,'keywords'].keys():
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
                    else:
                        fdf.loc[j,'keywords'][kr['fileclass']]=list()
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
        return bias
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
                    if kr['fileclass'] in fdf.loc[j,'keywords'].keys():
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
                    else:
                        fdf.loc[j,'keywords'][kr['fileclass']]=list()
                        fdf.loc[j,'keywords'][kr['fileclass']].append(kr['keyword'])
        return bias

    def get_classify_result(fdf,kdf,suspkdf,notification_corelation_dict):
        fdf["keywords"]=fdf["filename"].apply(lambda x:{})
        fdf["pred_class"]=fdf["filename"].apply(lambda x:list())
        fdf["final_categ"]=fdf["filename"].apply(lambda x:list())
        fdf.loc[fdf['filetype']=='TICKET','text_response']=fdf['table_response']
        classlis=[]
        for j,row in fdf.iterrows():
            if row['filetype']=='TICKET':
                text=str(''.join(unidecode.unidecode(row['text_response']).split()).lower())
            else:
                text=str(''.join(row['text_response'].split()))

            for i,kdrow in kdf.iterrows():
    #         #       
                if text[:5]!='Error':

                    f=True
                
                    for k in kdrow['keyword']:
                        if(kdrow['fileclass']=='N2' and k=='sucesion procesal'):
                                print(kdrow)
                        if not (str(''.join(unidecode.unidecode(k).split()).lower()) in text):
                            
                            f=False
                    if f and (kdrow['filetype']==row['filetype'] or(kdrow['filetype']=='NOTIFICATION' and row['filetype']=='OTHER') )and (kdrow['purpose']=='CLASSIFICATION'):
                        
                        fdf.loc[j,'pred_class'].append(Document_Analysis.get_predclass_normal(kdf,kdrow,j,text,fdf))
                        if kdrow['fileclass'] in fdf.loc[j,'keywords'].keys():
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
                            fdf.loc[j,'pred_class'].append(Document_Analysis.get_predclass_fuzzy(kdf,kdrow,j,text,fdf))
                            if kdrow['fileclass'] in fdf.loc[j,'keywords'].keys():
                                fdf.loc[j,'keywords'][kdrow['fileclass']].append(kdrow['keyword'])
                            else:
                                fdf.loc[j,'keywords'][kdrow['fileclass']]=list()
                                fdf.loc[j,'keywords'][kdrow['fileclass']].append(kdrow['keyword'])
            fdf.set_value(j,'pred_class',list(set(fdf.loc[j,'pred_class'])))
            for si,sr in suspkdf.iterrows():
                    f=True
                    for k in sr['keyword']:

                        if not (''.join(unidecode.unidecode(k).split()).lower() in ''.join(row['text_response'].split() )):
                            f=False
                    if f and (row['filetype']=='NOTIFICATION' or row['filetype']=='OTHER'):
                        if sr['remove_class'] in row['pred_class']:
                            if kdrow['fileclass'] in fdf.loc[j,'keywords'].keys():
                                fdf.loc[j,'keywords']['NX-'+sr['remove_class']].append(sr['keyword'])
                            else:
                                fdf.loc[j,'keywords']['NX-'+sr['remove_class']]=list()
                                fdf.loc[j,'keywords']['NX-'+sr['remove_class']].append(sr['keyword'])
                            fdf.loc[j,'pred_class'].remove(sr['remove_class'])
            f1=True
            print(row['pred_class'],fdf)
            for k in list(row['pred_class']): 
                s=set(notification_corelation_dict[k])
                if not(set(row['pred_class'])<=(s) and set([k])<=set(row['pred_class'])):

                    f1=False
            if f1:
                fdf.set_value(j,'final_categ',list(set(fdf.loc[j,'pred_class'])))
        fgs=fdf.groupby('filegroup')
        for k,v in fgs.groups.items():
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
    def filegroupAnalysis(ddf,fgdf):
        fgs = ddf.groupby('filegroup')
        i=0
        fgdf['predicted_classes'] = np.empty((len(fgdf), 0)).tolist()
        for k,v in fgs.groups.items():
            fgdf.loc[i,'filegroup']=k
            pcs=[]
            for ind in v:
                    pcs.append(ddf.loc[ind,'predicted_class'])
            fgdf.loc[i,'predicted_classes']+=pcs
            i+=1
        # y_actufg=[]
        y_predfg, conflic= [], []
        for k,v in fgs.groups.items():
        #     y_actufg.append(ddf.loc[v[0],'fileclass'])
            s=set()
            e='_alpha'
            for i in v:
                s.add(ddf.loc[i,'predicted_class'])
        #     s={x for x in s if x==x}
            if(len(s)==1):
                if list(s)[0]==list(s)[0]:
                    e=list(s)[0]
                    y_predfg.append(list(s)[0])
            else:
                if(len({x for x in s if x==x})>1):
                    e='Conflict'
                else:
                    for x in list(s):
                        if x==x:
                            e=x
            if e!='_alpha' and e!='Conflict':
                fgdf.loc[(fgdf['filegroup'] == k),'group_predicted_class']=e
            elif e=='Conflict':
                for j in v:
                    if ddf.loc[j,'filetype']=='TICKET':
                        e=ddf.loc[j,'predicted_class']
                        fgdf.loc[(fgdf['filegroup'] == k),'group_predicted_class']=e
        return fgdf


    def extract_data_from_filegroups(fdf,kdf):
        procedures = {'CAM':'Cambiario','COG':'Cognicion','CON':'Concurso Acreedores','EJE':'Juicio Ejecutivo',\
                    'ETJ':'EJECUCION DE TITULOS JUDICIAL','ETJH':'ETJ Continuacion Hipotecario','ETN':'EJECUCION DE TITULOS NO JUDICIAL',\
                    'HIP':'EJECUCION Hipotecaria','MCU':'Menor Cuantia','MON':'Monitorio','ORD':'Ordinario',\
                    'TDO':'Terceria De Domino','TMD':'Terceria De Mejor Derecho','VER':'Verbal','PEN':'Penal','RAP':'Recurso De Apelacion',\
                    'PTC':'Pieza Tasacion Costas','PSO':'Pieza Seperada Oposicion','AUX':'Auxilio Nacional','CNJ':'Cosignacion Judicial',\
                    'RCS':'Recurso De Casacion','INC':'Incidente Concursal','MC':'Medidas Cautelares','CN':'Conciliacion'}
        numspan = {"1":"uno","2":"dos","3":"tres","4":"cuatro","5":"cinco","6":"seis","7":"siete","8":"ocho","9":"nueve",\
                 "10":"diez","11":"once","12":"doce","13":"trece","14":"catorce","15":"quince","16":"dieciseis",\
                 "17" :"diecisiete","18":"dieciocho","19":"diecinueve","20" : "veinte","21" : "veintiuno","22"  :"veintidós",\
                 "23" : "veintitrés","24" : "veinticuatro","25" : "veinticinco","26":"veintiséis","27" :"veintisiete",\
                 "28": "veintiocho","29"  :"veintinueve","30" : "treinta"}
        numcat = {"1":"un","2":"dos","3":"tres","4":"quatre","5":"cinc","6":"sis","7":"set","8":"vuit","9":"nou",\
                "10":"deu","11":"onze","12":"dotze","13":"tretze","14":"catorze","15":"quinze","16":"setze",\
                "17" :"disset","18":"divuit","19":"dinou","20" : "vint"}
        extKeywords = kdf[kdf['purpose']=='EXTRACTION']
        extKeywords['decision_type'] = extKeywords['decision_type'].apply(lambda x : x.split('-')[1])
        fgdf = pd.DataFrame(columns=['filegroup'])
        ddf = fdf
        fgs = ddf.groupby('filegroup')
        i = 0
        fgdf['filetypes'] = np.empty((len(fgs.groups), 0)).tolist()
        fgdf['files'] = np.empty((len(fgs.groups), 0)).tolist()
        fgdf['predicted_classes'] = np.empty((len(fgs.groups), 0)).tolist()
        for k,v in fgs.groups.items():
            pcs, files, filetypes=[], [], []
            for ind in v:
                    files.append(ddf.loc[ind,'filename'])
                    filetypes.append(ddf.loc[ind,'filetype'])
                    pcs+=(ddf.loc[ind,'final_categ'])
        #             if len(ddf.loc[ind,'final_categ'])>0:
        #                 pcs.append(ddf.loc[ind,'final_categ'])
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
        fgdf['Court']=''
        fgdf['Solictor']=''
        c=0
        for fi,fr in fgdf.iterrows():
            auto=""
            match=None
            match1=set()
            match2=set()
            for i,r in fdf[(fdf['filetype']=='NOTIFICATION')&(fdf['filename'].isin(fr['files']))].iterrows():
                s= r['text_response']
                if len(s.strip())>0:
                    s=unidecode.unidecode('\n'.join(list(filter(None,s.split('\n')))).lower())
                    debtor=''
                    if 'contra' in s:
                            debtor=s.split('contra')[1].split('\n')[0]
                    elif 'parte demandada/ejecutada:' in s:
                            debtor=s.split('parte demandada/ejecutada:')[1].split('\n')[0]
                    if len(debtor.split(':'))>1:
                        fgdf.loc[fi,'Debtor']=debtor.split(':')[1].upper()
                    else:
                        fgdf.loc[fi,'Debtor']=debtor.upper()
                    fgdf.loc[fi,'Court']=s.split('\n')[0]
                    if 'procurador' in s.lower():
                        fgdf.loc[fi,'Solictor']=list(filter(None,s[s.lower().index('procurador')+10:].split('\n')))[0]
                    ptype=""
                    for x in procedures.values():
                        if x.lower() in ' '.join(s.split()):
                            fgdf.loc[fi,'Procedure_Type']=x
                            ptype=x
                            break
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
            if('N1'in fr['predicted_classes'])or('N3'in fr['predicted_classes'])or('N4'in fr['predicted_classes'])\
            or('N11'in fr['predicted_classes']):
                for ki,kr in extKeywords[(extKeywords['decision_type'].str.contains('AMOUNT'))].iterrows():
                    for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']==kr['filetype'])].iterrows():
                        text=''.join(r['text_response'].split())
                        if r['text_response'][:5]!='Error':
                            f=True
                            t=text
                            for k in kr['keyword']:
                                k1=''.join(unidecode.unidecode(k).split()).lower()
                                if  k1 in t:
                                    t=t[t.index(k1)+len(k1):]
                                else:
                                    f=False
                            k1=' '.join(kr['keyword'][0].split())    
                            if f :
                                tex=' '.join(r['text_response'].split())

                                fgdf.loc[fi,'Amount']=tex[tex.index(k1)+len(k1):].split()[0]
            if(('N9'in fr['predicted_classes'])or('N10'in fr['predicted_classes'])):
                    for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']=='NOTIFICATION')].iterrows():
                        text=' '.join(unidecode.unidecode(r['text_response']).split())
                        nls, nlc= [], []
                        if r['text_response'][:5]!='Error':
                            f=True
                            t=text
                            k1='dias'
                            if  k1 in t.lower():
                                nls+=  [t.lower().split()[i-1] for i, x in enumerate(t.lower().split()) if 'dias' in x ]
                            elif  'dies' in t.lower():
                                nlc+=  [t.lower().split()[i-1] for i, x in enumerate(t.lower().split()) if 'dies' in x ]
                            else:
                                    f=False
                            if f :
                                min=100
                                ls=[]
                                if len(nls)>0:
                                    fgdf.at[fi,'Numlist']=nls
                                    ls=nls
                                    numbers=numspan
                                elif len(nlc)>0:
                                    fgdf.at[fi,'Numlist']=nlc
                                    ls=nlc
                                    numbers=numcat
                                for num in ls:
                                    if num in numbers.values():
                                        n=int([x for x in numbers.keys() if numbers[x]==num ][0])
                                        if n < min:
                                            min=n
                                    elif num in numbers.keys():
                                        if int(num) < min:
                                            min=int(num)
                                if min!=100:
                                    fgdf.loc[fi,'Time Frame']=min


            for i,r in fdf[(fdf['filetype']=='TICKET')&(fdf['filename'].isin(fr['files']))].iterrows():
                    if r.table_response[:5]!='Error':
                        try:
                            js= json.loads(unidecode.unidecode( json.dumps(json.loads(r.table_response), ensure_ascii=False)))
                        except Exception as e:
                            js=json.loads(r.table_response)
                        table2=json.loads(js['table_2'])
        #                 print(r['filename'],js["table_1"][[x for x in js["table_1"].keys() if 'Fecha' in x][0]],table2['1'][[x for x in table2['1'].keys() if 'Fecha' in x][0]])
                        c+=1
                        fgdf.loc[fi,"Document date"]=js["table_1"][[x for x in js["table_1"].keys() if 'Fecha' in x][0]]
                        fgdf.loc[fi,"Stamp date"]=table2['1'][[x for x in table2['1'].keys() if 'Fecha' in x][0]]
        #                 print js
                        try:
                            fgdf.loc[fi,'Solictor']=js["table_1"]['Destinatarios']['_value']
                            fgdf.loc[fi,'Court']=js["table_1"]['Remitente'][[x for x in js["table_1"]['Remitente'].keys()\
                                                                             if str(x)[:6].lower()=='organo'][0]]
                        except Exception as e:
                            c+=1
                        auto=""
                        try:
                            s=json.loads(unidecode.unidecode(r['table_response']))['table_1']['Datos del mensaje']['Procedimiento destino']
                        except Exception as e:
                            s=str( json.loads(r['table_response'])['table_1']['Datos del mensaje'])
                        match1=set(re.findall(r'(\d{1,20}/\d{4})',s))
                        match2={'/'.join(x.split('/')[1:])for x in re.findall(r'(\d{1,2}/\d{1,2}/\d{4})',s)}
                        f=False
                        for x in procedures.keys():
                            if x.lower() in  s.lower():
                                f=True
                                proced= procedures[x]
                        if f:
                            fgdf.loc[fi,'Procedure_Type']=proced
                        if not (match1-match2) is None :
                            auto=list(match1-match2)[0]
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
                            for k,v in months.iteritems():
                                text=text.replace(k,v)
                            f=True
                            matches=datefinder.find_dates(text)
                            for match in matches:
                                fgdf.loc[fi,"Date_of_hearing"]=str(match)
                                break
            fgdf.loc[fi,'Auto']=auto
        return fgdf

    def insert_mongo(pdf_path,filename):
        mdb = DbConf.mdb
        fileData = DbConf.fileData
        # read in the pdf file.
        pdf_file = open(pdf_path, "rb");
        pdf_data = pdf_file.read()
        fs = GridFS(mdb)
        pdf_data = fs.put(pdf_data, filename=filename.split('.')[0])

        # read in the image.
        imgpath=join(ConfigClass.UPLOAD_FOLDER, filename)
    #     copyfile(pdf_path, imgpath)
        try:
            os.system("cd " + ConfigClass.UPLOAD_FOLDER + " && pdftoppm \"" + filename + "\" main -png")
            pgnum = get_pgnum(filename)
            ls=[]
            for i in range(0, pgnum):
                if (pgnum > 10 and i + 1 < 10):
                    ls.append(ConfigClass.UPLOAD_FOLDER + "/main-0" + str(i + 1) + ".png")
                else:
                    ls.append(ConfigClass.UPLOAD_FOLDER + "/main-" + str(i + 1) + ".png")
            imgs = [PIL.Image.open(i) for i in ls]
            min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
            imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
            imgs_comb = PIL.Image.fromarray(imgs_comb)
            imgs_comb.save(ConfigClass.UPLOAD_FOLDER + "/" + filename.rsplit('.', 1)[0] + '.jpg')
            image_file = open(imgpath, "rb");
            image_data = image_file.read()
            fs = GridFS(mdb)
            img_data = fs.put(image_data, filename=filename.split('.')[0])
        except Exception as e:
            img_data=str(e)
        data = { "filename":filename,
                "image_file": img_data,
                "actualfile": pdf_data,
               }
        mongo_id = fileData.insert_one(data)
        return mongo_id


    def read_pdf_n_insert(root_new,root_archive,model):
        PDF_DIR = root_new
        print("hii",root_new)
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
            flist=list()
            for k,gdf in  fg:
                fglist=[]
                elist=[]
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

            for i, r in fdf.iterrows():
                ticresponse=""
                textresponse=""
                try:
                    ticresponse=Document_Analysis.parse_ticket(join(PDF_DIR,r.filename))
                except Exception as e:
                    ticresponse='Error:'+str(e)
                try:
                    if r.filename[-3:].lower()!='zip':
                        if r.filetype=='TICKET':
                                textresponse=ticresponse
                        else:
                            textresponse=Document_Analysis.parse_other(join(PDF_DIR,r.filename))
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

                except Exception as e:
                    textresponse='Error:'+str(e)
                fdf.loc[i,"table_response"] = ticresponse
                fdf.loc[i,"text_response"] = textresponse
            fdf,fgdf=Document_Analysis.update_filetype(fdf)
            ############
            #mongoupdate here
            ############

            #######
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
            fdf=Document_Analysis.get_classify_result(fdf,kdf,suspkdf,notification_corelation_dict)

            fgdf=Document_Analysis.extract_data_from_filegroups(fdf,kdf)
            engine = create_engine(ConfigClass.SQLALCHEMY_DATABASE_URI)
            session = Document_Analysis.loadSession(engine)
            max_v = model.db.session.query(model.db.func.max(model.ProccessLog.batch_id)).scalar()
            if max_v==None:
                newmax=1
            else:
                newmax=max_v+1
            proccess_log = model.ProccessLog( batch_id=newmax,
                                            creation_date = datetime.datetime.now(),process_date=datetime.datetime.now())
            model.db.session.add(proccess_log)
            model.db.session.commit()
            fgdf=fgdf.applymap(lambda x: None if x=='' else x)
            for i , rr in fgdf.iterrows():
                kk = model.FileGroup(file_group = rr['filegroup'],
                                               court = rr['Court'],
                                               court_initial = rr['Court'],
                                               solicitor = rr['Solictor'],
                                               solicitor_initial = rr['Solictor'],
                                               procedure_type =rr['Procedure_Type'],
                                               procedure_type_initial = rr['Procedure_Type'],
                                               time_frame = rr['Time Frame'],
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
                                               debtor = rr['Debtor'],
                                               batch_id=newmax,
                                               creation_date = datetime.datetime.now()
                                    )
                model.db.session.add(kk)
                model.db.session.commit()
            for i , r in fdf.iterrows():
                k = model.FileClassificationResult(file_name =r['filename'],
                                             file_group =r['filegroup'],
                                             file_type=r['filetype'],
                                             predicted_classes=json.dumps(r['final_categ']),
                                             batch_id=newmax,
                                             creation_date = datetime.datetime.now())
                model.db.session.add(k)
                model.db.session.commit()
                shutil.move(join( PDF_DIR,r.filename),join(root_archive,r.filename))
            return True
        else:
            return False
        
