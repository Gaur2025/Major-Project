from flask import Flask,render_template,request
import numpy as np
import pickle

model=pickle.load(open('random_forest.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def new():
    return render_template('new.html')

@app.route('/predict', methods=['POST','GET'] )
def predict():
    """
    features = ['setting1', 'setting2', 's2', 's3', 's4',
                 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15',
                 's17', 's20', 's21',
                  'av2', 'av3', 'av4', 'av7', 'av8', 'av9',
                 'av11', 'av12', 'av13', 'av14', 'av15', 'av17',
                 'av20', 'av21', 'sd2', 'sd3', 'sd4', 'sd7', 'sd8',
                 'sd9', 'sd11', 'sd12', 'sd13', 'sd14', 'sd15', 'sd17',
                 'sd20', 'sd21']
    """
    setting1=float(request.form['setting1'])  # [-0.008700, 0.008700]
    setting2=float(request.form['setting2'])  # [-0.000600, 0.000600]

    # s2=float(request.form['s2'])
    # s3=float(request.form['s3'])   
    # s4=float(request.form['s4'])
    # s7=float(request.form['s7'])
    # s8=float(request.form['s8'])
    # s9=float(request.form['s9'])
    # s11=float(request.form['s11'])
    # s12=float(request.form['s12'])
    # s13=float(request.form['s13'])
    # s14=float(request.form['s14'])
    # s15=float(request.form['s15'])
    # s16=float(request.form['s16'])
    # s17=float(request.form['s17'])
    # s20=float(request.form['s20'])
    # s21=float(request.form['21'])

    # av2=float(request.form['av2'])
    # av3=float(request.form['av3'])   
    # av4=float(request.form['av4'])
    # av7=float(request.form['av7'])
    # av8=float(request.form['av8'])
    # av9=float(request.form['av9'])
    # av11=float(request.form['av11'])
    # av12=float(request.form['av12'])
    # av13=float(request.form['av13'])
    # av14=float(request.form['av14'])
    # av15=float(request.form['av15'])
    # av17=float(request.form['av17'])
    # av20=float(request.form['av20'])
    # av21=float(request.form['av21'])

    sd2=float(request.form['sd2'])
    sd3=float(request.form['sd3'])   
    sd4=float(request.form['sd4'])
    sd7=float(request.form['sd7'])
    sd8=float(request.form['sd8'])
    sd9=float(request.form['sd9'])
    sd11=float(request.form['sd11'])
    sd12=float(request.form['sd12'])
    sd13=float(request.form['sd13'])
    sd14=float(request.form['sd14'])
    sd15=float(request.form['sd15'])
    sd17=float(request.form['sd17'])
    sd20=float(request.form['sd20'])
    sd21=float(request.form['sd21'])
    
    
    # features=np.array([setting1,setting2, s2, s3, s4,  s7, s8, s9, 
    # s11, s12, s13, s14, s15, s16, s17,  s20, s21,
    #  av2, av3, av4,  av7, av8, av9, 
    # av11, av12, av13, av14, av15,  av17,  av20, av21, sd2, sd3, sd4,  sd7, sd8, sd9, 
    # sd11, sd12, sd13, sd14, sd15, sd17, sd20, sd21,
    # ])

    features=np.array([setting1,setting2, 
    sd2, sd3, sd4,  sd7, sd8, sd9, 
    sd11, sd12, sd13, sd14, sd15, sd17, sd20, sd21,
    ])
    pred = model.predict([features])
    
    def statement():
        if pred == 0:
            return 'Result:- The model has predicted that engine will FAIL.'
        elif pred == 1:
            return 'Result:- The model has predicted that engine will NOT FAIL.'
    
    return render_template('new.html',statement=statement())


if __name__=='__main__':
    app.run()