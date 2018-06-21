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
import psycopg2
import hashlib
import ast
import json
import unicodedata
import time
import PIL
from gridfs import GridFS
from PIL import Image
from shutil import copyfile
import psycopg2
from sqlalchemy import create_engine
import multiprocessing as mp
from configuration.configuration import ConfigClass,PyDbLite,DbConf


def keywordimport():
    engine = create_engine(ConfigClass.SQLALCHEMY_DATABASE_URI)
    keyword_df= pd.read_sql_query('SELECT k.id,fc.file_class_name as fileclass,ft.type,p.purpose_type,\
    k.decision_type,k.keyword_list,bia.file_class_name as bias FROM keywords k left Join file_class fc \
    on fc.id = k.file_class_id left Join file_class bia on bia.id = k.bias left Join file_class f \
    on f.id = k.file_class_id left Join file_types ft on ft.id = k.file_type_id left Join purpose p \
    on p.id = k.purpose_id',engine)
#         with open('Final_Keyword_Analysis.pickle', 'rb') as handle:
#             b=pickle.load(handle)
#         kxddf=b['keywordsX']
#         kxddf.drop(173,inplace=True)
    #copy1 postgres copy2 pydblite
    pyDbLite_db = PyDbLite().pyDbLite_db
    pyDb_id = pyDbLite_db.insert(keyword_df=keyword_df)
    keyword_df=keyword_df.rename(columns={"keyword_list": "keyword","type":"filetype","purpose_type":"purpose"})
    keyword_df['keyword']=keyword_df['keyword'].apply(lambda x:json.loads(x))
    return keyword_df, pyDb_id


def get_pgnum(filename):
    pdf = pdfquery.PDFQuery(ConfigClass.UPLOAD_FOLDER + "/" + filename)
    pdf.load()
    pgn = len(pdf.tree.getroot().getchildren())
    return pgn


def get_structured_files_dataframe(df):
    tmp = pd.DataFrame(columns=["file","file_id","group"])
    j = 0
    for i, r in df.iterrows():
        for f in r.files:
            tmp.loc[j] = [f,r.file_ids[r.files.index(f)], r.group]
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
    print('unrtf --text '+path)
    text = os.popen('unrtf --text '+path).read()
    print(text)
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
                    [box(*[float(x) for x in node.get("bbox")[1:-1].split(",")])
                     for node in root.iter() if node.tag == "LTRect"]
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
            return('Error:'+str(e))

    return json.dumps(_JSON, ensure_ascii=False)

def parse_other(pf):
    try:
        newtext =str(textract.process(pf),'utf-8')
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

    return text

def update_filetype(fdf):
    fgs=fdf.groupby('filegroup')
    fgdf=pd.DataFrame(columns=['filegroup'])
    i=0
    fgdf['files']=np.empty((len(fgs.groups), 0)).tolist()
    fgdf['file_ids']=np.empty((len(fgs.groups), 0)).tolist()
    fgdf['filetypes']=np.empty((len(fgs.groups), 0)).tolist()
    for k,v in fgs.groups.items():

        fgdf.loc[i,'filegroup']=k
        files=[]
        pcs=[]
        filetypes=[]
        file_ids=[]
        for ind in v:
            files.append(fdf.loc[ind,'filename'])
            filetypes.append(fdf.loc[ind,'filetype'])
            file_ids.append(fdf.loc[ind,'file_id'])
        fgdf.loc[i,'filetypes']=filetypes
        fgdf.loc[i,'files']=files
        fgdf.loc[i,'file_ids']=file_ids
        i+=1
    print(fdf['table_response'])
    for i ,r in fdf[~(fdf['table_response'].str.contains('Error:'))&(fdf['filetype']=='TICKET')].iterrows():
        if r['table_response'][:5]!='Error':
            js=json.loads(r['table_response'])
            pf=r['filename'].split('.')[0]+'_'+''.join(js['table_1']['Documentos'][0][0].split()).split('(Principal)')[0]
            fgf=[''.join(x.split())for x in fgdf.loc[fgdf['filegroup']==r['filegroup'],'files'].values[0] ]
            if pf in fgf:
                fl=fgdf.loc[fgdf['filegroup']==r['filegroup'],'files'].values[0][fgf.index(pf)]
                fdf.loc[fdf['filename'].str.contains(fl),'filetype']="NOTIFICATION"
    return fdf,fgdf

def fileAnalysis(fdf,kdf):
    kdf["document_list"] = np.empty((len(kdf), 0)).tolist()
    fdf["keywords"] = fdf["filename"].apply(lambda x:{})
    classlist = list()
    notextset = set()
    for j, row in fdf.iterrows():
        if j%595 == 0:
            print(j)
        text = ''.join(row['text_response'].split())
        
        for i, kdrow in kdf.iterrows():
            if row['text_response'][:5] != 'Error':
                f=True
                for k in kdrow['keyword']:
                    if not str((''.join(unidecode.unidecode(k).split()).lower())) in text:
                        f=False
                   
                if f and (kdrow['filetype']==row['filetype'])and (kdrow['purpose']=='CLASSIFICATION'):
                    kdf.loc[i,"document_list"].append(row['file_id'])
                    if kdrow['fileclass'] in fdf.loc[j,'keywords'].keys():
                        fdf.loc[j,'keywords'][kdrow['fileclass']].append(unidecode.unidecode(str(kdrow['keyword'])))
                    else:
                        fdf.loc[j,'keywords'][kdrow['fileclass']] = list()
                        fdf.loc[j,'keywords'][kdrow['fileclass']].append(unidecode.unidecode(str(kdrow['keyword'])))
        for ki, kr in kdf.iterrows():
            c=0
            if row['file_id'] in kr['document_list'] :
                c=1
                if not kr['fileclass']==kr['bias'] :
                    #c=kr[kr['file_class']]
                    c=0
                if kr['fileclass'] in fdf.columns:
                    fdf.loc[j,kr['fileclass']]+= c
                else:
                    fdf[kr['fileclass']] = 0
                    classlist.append(kr['fileclass'])
                    fdf.loc[j,kr['fileclass']] = c
    print("Part2")
    for i, row in fdf.iterrows():
        if(len(classlist)>0):
            keyscore = dict(row[classlist])
        else:
            keyscore = {'Nan':0}
        predicted = max(keyscore, key=keyscore.get)
        if 'N5' in row.to_dict().keys() and row['N5']>0:
            fdf.loc[i,'predicted_class'] = 'N5'
        elif set(['N2','N4'])<= set(fdf.columns) and row['N2']>0 and row['N4']>0:
            fdf.loc[i,'predicted_class'] = 'N2+N4'
        elif set(['N1','N9','N10'])<= set(fdf.columns) and row['N1']>0 and (row['N10']>0 or row['N9']>0):
            fdf.loc[i,'predicted_class'] = 'N1'
        elif set(['N6','N2','N10'])<= set(fdf.columns) and row['N10']>0 and (row['N2']>0 or row['N6']>0):
            fdf.loc[i,'predicted_class'] = 'N2+N4'
        elif keyscore[predicted]>0:
            fdf.loc[i,'predicted_class'] = predicted
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


def get_textExtract(fgdf,fdf,kdf):
    procedures={'CAM':'Cambiario','COG':'Cognicion','CON':'Concurso Acreedores','EJE':'Juicio Ejecutivo',
                'ETJ':'EJECUCION DE TITULOS JUDICIAL','ETJH':'ETJ Continuacion Hipotecario','ETN':'EJECUCION DE TITULOS NO JUDICIAL',\
                'HIP':'EJECUCION Hipotecaria','MCU':'Menor Cuantia','MON':'Monitorio','ORD':'Concurso Ordinario',
                'TDO':'Terceria De Domino','TMD':'Terceria De Mejor Derecho','VER':'Verbal','PEN':'Penal','RAP':'Recurso De Apelacion',\
                'PTC':'Pieza Tasacion Costas','PSO':'Pieza Seperada Oposicion','AUX':'Auxilio Nacional','CNJ':'Cosignacion Judicial',
                'RCS':'Recurso De Casacion','INC':'Incidente Concursal','MC':'Medidas Cautelares','CN':'Conciliacion'}
    numspan={"1":"uno","2":"dos","3":"tres","4":"cuatro","5":"cinco",\
             "6":"seis","7":"siete","8":"ocho","9":"nueve","10":"diez",\
             "11":"once","12":"doce","13":"trece","14":"catorce","15":"quince",\
             "16":"dieciseis","17" :"diecisiete","18":"dieciocho","19":"diecinueve",\
             "20" : "veinte","21" : "veintiuno","22"  :"veintidos","23" : "veintitres","24" : "veinticuatro","25" : "veinticinco",\
             "26":"veintiseis","27" :"veintisiete","28": "veintiocho","29"  :"veintinueve","30" : "treinta"}
    numcat={"1":"un","2":"dos","3":"tres","4":"quatre","5":"cinc","6":"sis","7":"set","8":"vuit","9":"nou","10":"deu",
            "11":"onze","12":"dotze","13":"tretze","14":"catorze","15":"quinze","16":"setze","17" :"disset","18":"divuit",
            "19":"dinou","20" : "vint"}
    extKeywords=kdf[kdf['purpose']=='EXTRACTION']
    extKeywords['decision_type']=extKeywords['decision_type'].apply(lambda x : x.split('-')[1])
    c=0
    fgdf['Numlist']=[[] for _ in range(len(fgdf))]
    fgdf['Time Frame']=''
    fgdf['Amount']=''
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
        if(fr['group_predicted_class']=='N1')or(fr['group_predicted_class']=='N3')or(fr['group_predicted_class']=='N4'):
            for ki,kr in extKeywords[(extKeywords['decision_type'].str.contains('AMOUNT'))&(extKeywords['fileclass']==fr['group_predicted_class'])].iterrows():
                for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']==kr['filetype'])].iterrows():
                    text=''.join(r['text_response'].split())
        #         
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
        if((fr['group_predicted_class']=='N9')or(fr['group_predicted_class']=='N10')):
    #         for ki,kr in extKeywords[(extKeywords['notification_type'].str.contains('TIME FRAME'))&(extKeywords['fileclass']==fr['group_predicted_class'])].iterrows():
                for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']=='NOTIFICATION')].iterrows():
                    text=' '.join(unidecode.unidecode(r['text_response']).split())
        #           
                    nls=[]
                    nlc=[]
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

        if (fr['group_predicted_class']=='N5')|(fr['group_predicted_class']=='N6')|(fr['group_predicted_class']=='N9'):


            for ki,kr in extKeywords[~(extKeywords['decision_type'].str.contains('TIME FRAME'))&(extKeywords['fileclass']==fr['group_predicted_class'])].iterrows():
                for i,r in fdf[(fdf['filename'].isin(fr['files']))&(fdf['filetype']=='NOTIFICATION')].iterrows():
                    text=''.join(r['text_response'].split())
        #         
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
                            fgdf.loc[fi,fr['group_predicted_class']+'-Extraction']=unidecode.unidecode(str([dict({kr['notification_type']:unidecode.unidecode(str(kr['keyword']))})]))
                            if fr['group_predicted_class']=='N5':
                                fgdf.loc[fi,'Time Frame']=5
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
                        fgdf.loc[fi,'Court']=js["table_1"]['Remitente'][[x for x in js["table_1"]['Remitente'].keys() if str(x)[:6].lower()=='organo'][0]]

                    except Exception as e:
                        c+=1
                    auto=""
                    try:
                        s=json.loads(unidecode.unidecode(r['table_response']))['table_1']['Datos del mensaje']['Procedimiento destino']

                    except Exception as e:
                        s=str( json.loads(r['table_response'])['table_1']['Datos del mensaje'])
            #             print str(e)
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

    # read in the image.Racmo_backendconfiguration
    imgpath=join(ConfigClass.UPLOAD_FOLDER, filename.split('.')[0]+'.png')
    copyfile(pdf_path, imgpath)
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
        imgs_comb.save(imgpath)
        
        image_file = open(imgpath, "rb");
        image_data = image_file.read()
        fs = GridFS(mdb)
        img_data = fs.put(image_data, filename=filename.split('.')[0])
    except Exception as e:
        image_file = open("/home/thrymr/Racmo/pro/unparsed.png", "rb");
        image_data = image_file.read()
        fs = GridFS(mdb)
        img_data = fs.put(image_data, filename=filename.split('.')[0])        
    data = { "filename":filename,
            "image_file": img_data,
            "actualfile": pdf_data,
           }
    mongo_id = fileData.insert_one(data)
    os.remove(imgpath)
    return mongo_id.inserted_id

def read_pdf_n_insert(pdf_dir_root):
    PDF_DIR = pdf_dir_root
    pdf_files = [f for f in listdir(PDF_DIR) if isfile(join(PDF_DIR,  f))]
    ###################
    ## Save in mongo ##
    ###################
    ls=list()
    mongo_ids=dict()
    for pdf_file in pdf_files:
        # mongo_ids.update({pdf_file:insert_mongo(join(pdf_dir_root,pdf_file),pdf_file)})
        mongo_ids.update({pdf_file:1})
        ls.append(pdf_file.split('_'))
    df=pd.DataFrame(ls)
    # df1=df.groupby(3)
    df = df[pd.notnull(df[3])]
    fg=list(df.groupby(3))
    ls=[]
    i=0
    for k,gdf in  fg:
        fglist, elist, flist = [], [], []
        for i, row in gdf.iterrows():
            row = row.dropna()
            fln = '_'.join(list(row))
            flist.append(fln)
            elist.append(fln[-3:].lower())
        fgroup={'group':k,'files':flist,'length':len(flist),'min_filename':min(flist, key=len),'extensions':elist}
        
        #save fgroup in postgres in filegroup#
        # conn = psycopg2.connect(database=DbConf.name, user=DbConf.username,
        #                         password=DbConf.password,host=DbConf.host, port=DbConf.port)
        # # insert data in tag Table
        # cur = conn.cursor()
        # cur.execute("insert into file_group (file_group_name) values('" + k + "') RETURNING id;")
        # fg_id = cur.fetchone()[0]
        fglist.append(1)

#         fg=inserpg_filegroup(fgroup)
        ####
        fidlis=list()

        for f in flist:
            file_data_id=mongo_ids[f]

            ##save file in postgres
#             fidlis.append(insertpg_file(f,fg))
#             cur1 = conn.cursor()

#             cur1.execute("insert into file_info (file_group_id,file_data_id,file_name) values(" +str(fg_id)+",'"+str(file_data_id)+"','"+f+"') RETURNING id;")
#             f_info_id = cur1.fetchone()[0]
            fidlis.append(i)
            i+=1
            ##
        # conn.commit()
        fgroup['file_ids']=fidlis
        ls.append(fgroup)
    flgdf=pd.DataFrame(ls)
    flgdf=flgdf.dropna(thresh=1,axis=1)
    fdf = get_structured_files_dataframe(flgdf)
    fdf=fdf.rename(columns={"file":"filename","group":"filegroup","type":"filetype"})

    for i, r in fdf.iterrows():
        ticresponse=""
        textresponse=""
        try:
            ticresponse=parse_ticket(join(PDF_DIR,r.filename))
        except Exception as e:
            ticresponse='Error:'+str(e)
        try:
            textresponse=parse_other(join(PDF_DIR,r.filename))
        except Exception as e:
            textresponse='Error:'+str(e)
        fdf.loc[i,"table_response"] = ticresponse
        fdf.loc[i,"text_response"] = textresponse
    fdf,fgdf=update_filetype(fdf)
    #######
    #kdf from pg admin and keyword analysis from pydblite
    kdf,_=keywordimport()
    # fdf=fileAnalysis(fdf,kdf)
    pool = mp.Pool(processes=4)
    ind=int()
    results = [pool.apply(fileAnalysis, args=(fdf[int(x*len(fdf)/10):int((x+1)*len(fdf)/10)],kdf,)) for x in range(0,10)]
    fdf=pd.concat(results)
    fgdf=filegroupAnalysis(fdf,fgdf)
    PvRdf=get_textExtract(fgdf,fdf,kdf)
    print(fdf)
#     ##############
#     ###Savepydblite  fdf fgdf with path and date 
#     ##############
#     ##fdf fgdf

#     pyDbLite_db = PyDbLite().pyDbLite_db
#     pyDb_id = pyDbLite_db.insert(file_filegroup = {'path':pdf_dir_root,'file':fdf,'filegroup_result':PvRdf})
#     pyDbLite_db.commit()
#     ##resultdf

#     # pyDbLite_db = PyDbLite().pyDbLite_db
#     data=pyDbLite_db[1]
#     if not (data['result_df'] is None):
#         data['result_df'].append(PvRdf)
#         pyDb_id = pyDbLite_db.update(data,result_df=PvRdf)
#     else:
#         pyDb_id = pyDbLite_db.update(data,result_df=PvRdf)
#     pyDbLite_db.commit()
# #instertpydblite 1)with path and date 2) append to overall
#     writer = pd.ExcelWriter(PDF_DIR+'/'+str(time.time())+'File_Details.xlsx', engine='openpyxl')
#     fdf[['filename','filegroup','filetype','keywords']].to_excel(writer,'Sheet1')
#     writer.save()
#     writer = pd.ExcelWriter(PDF_DIR+'/'+str(time.time())+'File_Group_Result.xlsx', engine='openpyxl')
#     PvRdf.to_excel(writer,'Sheet1')
#     writer.save()
    return PvRdf

pyDbLite_db = PyDbLite().pyDbLite_db


if __name__ == '__main__': 
    read_pdf_n_insert("/home/thrymr/Racmo/processed/Test")

