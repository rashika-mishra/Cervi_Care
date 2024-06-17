from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
dataset=pd.read_csv('kag_risk_factors_cervical_cancer.csv')
x=dataset.iloc[:,0:35]
y=dataset.iloc[:,35]
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[[
'Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)','STDs','STDs (number)','STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV',
'STD SS','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology','Schiller','Citology']])
(x[[
'Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)','STDs','STDs (number)','STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV',
'STD SS','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology','Schiller','Citology']])=imputer.transform(x[[
'Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)','STDs','STDs (number)','STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV',
'STD SS','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology','Schiller','Citology']])
x=x.drop(columns=[ 'Smokes','Smokes (years)','IUD','Hormonal Contraceptives','STDs (number)', 'STDs:condylomatosis', 'STDs (number)', 'STDs:vulvo-perineal condylomatosis', 'Dx:HPV', 'Schiller.1', 'Citology.1'])
#splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

model=pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    data=''
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')
@app.route('/diagnose')
def home():
    return render_template('home.html')
@app.route('/faq')
def faq():
    return render_template('faq.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict',methods=['POST'])
def dignos():
    
    print("data recieved")
    print(request.form)
    data1=request.form['data1']
    data2=request.form['data2']
    data3=request.form['data3']
    data4=request.form['data4']
    data5=float(request.form['data5'])
    data6=float(request.form['data6'])
    data7=float(request.form['data7'])
    data8=float(request.form['data8'])
    data9=request.form['data9']
    data10=request.form['data10']
    data11=request.form['data11']
    data12=request.form['data12']
    data13=request.form['data13']
    data14=request.form['data14']
    data15=request.form['data15']
    data16=request.form['data16']
    data17=request.form['data17']
    data18=request.form['data18']
    data19=request.form['data19']
    data20=request.form['data20']
    data21=request.form['data21']
    data22=request.form['data22']
    data23=request.form['data23']
    data24=request.form['data24']
    data25=request.form['data25']
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23,data24,data25]])
    pred=model.predict(sc.transform(arr))
    print(pred)
    return render_template('home.html',data=pred)

if __name__ == '__main__':
    app.run(debug=True)
