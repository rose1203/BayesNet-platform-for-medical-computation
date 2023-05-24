# coding=gbk
from flask import Flask 
from flask import render_template
import bnlearn as bn
import pandas as pd
import numpy as np
from flask import request
import os
from werkzeug.utils import secure_filename
import chardet
from flask import jsonify
import csv

app = Flask(__name__) 
@app.route("/") 
def index(): 
  return  render_template("index.html")


@app.route("/learning",methods=["GET","POST"]) 
def learning():
    blacklist=[]
    score_value=[0,0,0,0]
    bl_checked=[]
    file_path=''
    
    if request.method == 'POST':
      if request.form['submit'] == 'LearnNet':      
          filename = request.form['filename']         
          data=pd.read_csv(filename)
          #data.fillna('unknown', inplace=True) 
          blacklist=request.form.getlist('columns')     
          nodes = list(data.columns)
          values = [data[col].unique().tolist() for col in data.columns]
          new_nodes = [element for element in nodes if element not in blacklist]
          dfhot, dfnum = bn.df2onehot(data)
          new_values = [values[i] for i in range(len(values)) if nodes[i] in new_nodes]          
           
          values_model = [dfnum[col].unique().tolist() for col in dfnum.columns]            
          new_values_model = [values_model[i] for i in range(len(values_model)) if nodes[i] in new_nodes]
          for i in range(len(new_values)):
              delimiter = ", "
              new_values[i] = delimiter.join(str(x) for x in new_values[i])
          for i in range(len(new_values_model)):
              delimiter = ", "
              new_values_model[i] = delimiter.join(str(x) for x in new_values_model[i])    
          print(new_nodes,new_values,new_values_model)
          nodes_df = pd.DataFrame({'nodes': new_nodes, 'values': new_values,'new_values_model':new_values_model})
          nodes_file = 'nodes.csv'
          
          nodes_df.to_csv(nodes_file, index=False)
          structure_methodtype=request.values.get("structure_methodtype")          
          para_methodtype=request.values.get("para_methodtype")          
          dfhot, dfnum = bn.df2onehot(data)
          
          if not blacklist is None:
              # Structure learning
              DAG = bn.structure_learning.fit(dfnum, methodtype=structure_methodtype,bw_list_method='nodes',black_list=blacklist)
             
          else:
              # Structure learning       
              DAG = bn.structure_learning.fit(dfnum, methodtype=structure_methodtype,black_list=blacklist,bl_checked=bl_checked)
            
          # Parameter learning          
          model = bn.parameter_learning.fit(DAG,dfnum,methodtype=para_methodtype)        
          bn.plot(DAG,'./static/assets/img/bnlearn_DAG.png',params_interactive={'layout':'shell_layout'})  
          scores=model['structure_scores']
          #score_key=[key for key in scores]
          score_value=[value for value in scores.values()]
          bn.save(model, filepath='./bnlearn_model', overwrite=True)         
    
      elif request.form['submit'] == 'SaveNet':
          text = request.form.get('savename')
          model = bn.load(filepath='./bnlearn_model')
          filepath = os.path.join('./model', text)
          bn.save(model, filepath=filepath, overwrite=True)
          data=pd.read_csv('nodes.csv')
          data.to_csv(os.path.join('./model/nodes', text+'.csv'), index=False, encoding='utf-8')
          
      elif request.form['submit'] == 'UploadData':
          file = request.files["csv_file"]          
          if file.filename != '':
         
              filename = secure_filename(file.filename)
              df = pd.read_csv(file)
              filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
              file.seek(0)
              file.save(filepath)
              columns = [{'name': column, 'checked': False} for column in df.columns]
              return render_template("learning.html",score_value=np.round(score_value,2), columns=columns,bl_checked=bl_checked,file_path=filepath)
              
    return  render_template('learning.html',score_value=np.round(score_value,2),bl_checked=bl_checked,file_path=file_path)
    

    
@app.route("/inference",methods=["GET","POST"]) 
def inference(): 
    inference=[0,0]
    evidence_dict={}
    directory_path = "./model/nodes"
    file_names = os.listdir(directory_path)
  
    file_list = []
    for file_name in file_names:
        file_list.append(os.path.splitext(file_name)[0])
    #file_list.insert(0, 'Please choose a model')
    print (file_list)
   
    if request.method=="POST":
    
        select_file= request.form['file-select']+'.csv'
        
        with open(os.path.join('./model/nodes/', select_file)) as csvfile:
            reader = csv.reader(csvfile)
            column = []        
            for row in reader:
                column.append(row[0])
        node_dict = dict.fromkeys(column)
        for key in node_dict:
            node_dict[key] = request.values.get(key) 
        
        for key,value in node_dict.items():
            if not value is None:
                evidence_dict[key]=int(value)
        
        
        select_file=os.path.splitext(select_file)[0]
        filepath=os.path.join('./model/', select_file)
        print(filepath)
        model = bn.load(filepath=filepath)
       
        target=request.form['content-select']
        
        # Make inference
        q = bn.inference.fit(model, variables=[target], evidence=evidence_dict)

        #print(q.df)
        query=bn.query2df(q, variables=[target])
        inference=[round(query.iloc[0,1],4),round(query.iloc[1,1],4)]

    return  render_template("inference.html",inference=inference, files=file_list)

@app.route('/get_file_data', methods=['POST'])
def get_file_data():
   
    file_name = request.form['file_name']+'.csv'
    with open(os.path.join('./model/nodes/', file_name), encoding='utf-8') as f:
        reader = csv.reader(f)
        data = [row[0] for row in reader]
        
    return jsonify({'data': data})

@app.route('/get_radios', methods=['POST'])
def get_radios():
    
    file_name = request.form['file_name']+'.csv'
    with open(os.path.join('./model/nodes/', file_name), encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    
    radios = '<table>'
    for option in data:
        lst = option[1].split(",")
        lst_model=option[2].split(",")
        radios+=f'<tr>  <td>{option[0]}</td><td>'
        for i in range(len(lst)):
            radios+=f'<input name="{option[0]}" value="{lst_model[i]}" type="radio" > {lst[i]}'
        radios+=f'</td> </tr>'
        
    radios += '</table>'
    print(radios)
    return jsonify({'header': header, 'radios': radios})


if __name__ == "__main__": 
  app.config['UPLOAD_FOLDER'] = './uploads'
  app.run(host='0.0.0.0',port =8010,debug=True)
  app.jinja_env.auto_reload = True
  
  app.config['TEMPLATES_AUTO_RELOAD'] = True