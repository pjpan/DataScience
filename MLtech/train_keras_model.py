from keras.regularizers import l2, activity_l2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.optimizers import Adagrad,SGD,Adadelta
from keras.callbacks import EarlyStopping
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.constraints import maxnorm

np.random.seed(1778)  # for reproducibility
need_normalise=True
need_validataion=True
need_categorical=False
save_categorical_file=False
nb_epoch=1#400
golden_feature=[("CoverageField1B","PropertyField21B"),
                ("GeographicField6A","GeographicField8A"),
                ("GeographicField6A","GeographicField13A"),
                ("GeographicField8A","GeographicField13A"),
                ("GeographicField11A","GeographicField13A"),
                ("GeographicField8A","GeographicField11A")]

def save2model(submission,file_name,y_pre):
    assert len(y_pre)==len(submission)
    submission['QuoteConversion_Flag']=y_pre
    submission.to_csv(file_name,index=False)
    print ("saved files %s" % file_name)

def getDummy(df,col):
    category_values=df[col].unique()
    data=[[0 for i in range(len(category_values))] for i in range(len(df))]
    dic_category=dict()
    for i,val in enumerate(list(category_values)):
        dic_category[str(val)]=i
   # print dic_category
    for i in range(len(df)):
        data[i][dic_category[str(df[col][i])]]=1

    data=np.array(data)
    for i,val in enumerate(list(category_values)):
        df.loc[:,"_".join([col,str(val)])]=data[:,i]

    return df


def generateFileName(model,params):
     file_name="_".join([(key+"_"+ str(val))for key,val in params.items()])
     return model+"_"+file_name+".csv"

def load_data():
    train=pd.read_csv('../input/train.csv')
    test=pd.read_csv('../input/test.csv')
    train = train.drop(['QuoteNumber','PropertyField6', 'GeographicField10A'], axis=1)

    submission=pd.DataFrame()
    submission["QuoteNumber"]= test["QuoteNumber"]

    train_y=train['QuoteConversion_Flag'].values
    train=train.drop('QuoteConversion_Flag',axis=1)

    test = test.drop(['QuoteNumber','PropertyField6', 'GeographicField10A'],axis=1)
    train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
    train = train.drop('Original_Quote_Date', axis=1)
    train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
    train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
    train['weekday'] = [train['Date'][i].dayofweek for i in range(len(train['Date']))]

    test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
    test = test.drop('Original_Quote_Date', axis=1)
    test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
    test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
    test['weekday'] = [test['Date'][i].dayofweek for i in range(len(test['Date']))]

    train = train.drop('Date', axis=1)
    test = test.drop('Date', axis=1)

    #fill na, have no effect on need_categorical method
    train = train.fillna(-1)
    test = test.fillna(-1)

    for f in test.columns:#
        if train[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(train[f])+list(test[f]))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

    #try to encode all params less than 100 to be category
    if need_categorical:
        #row bind train and test
        x=train.append(test,ignore_index=True)
        del train
        del test
        for f in x.columns:#
            category_values= set(list(x[f].unique()))
            if len(category_values)<4:
                print (f)
                x=getDummy(x,f)
                #x.drop(f,axis=1)
                #all_data.drop(f,axis=1)
        test = x.iloc[260753:,]
        train = x.iloc[:260753:,]

    #save need_categorical:
#    if save_categorical_file and need_categorical:
#        train["QuoteConversion_Flag"]=train_y
#        train["QuoteNumber"]=1
#        test["QuoteNumber"]=submission["QuoteNumber"]
#        train.to_csv("./data/train_category.csv",index=False)
#        test.to_csv("./data/test_category.csv",index=False)

    #add golden feature:

    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y).astype(np.int32)
    train_y = np_utils.to_categorical(train_y)
    #for featureA,featureB in golden_feature:
    #    train.loc[:,"_".join([featureA,featureB,"diff"])]=train[featureA]-train[featureB]
    #    test.loc[:,"_".join([featureA,featureB,"diff"])]=test[featureA]-test[featureB]

    print ("processsing finished")
    valid=None
    valid_y=None
    train = np.array(train)
    train = train.astype(np.float32)
    test=np.array(test)
    test=test.astype(np.float32)
    if need_normalise:
        scaler = StandardScaler().fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

    if need_validataion:
        train,valid,train_y,valid_y=train_test_split(train,train_y,test_size=20000,random_state=218)
    return [(train,train_y),(test,submission),(valid,valid_y)]

print('Loading data...')

#datasets=load_data()

datasets=load_data()

X_train, y_train = datasets[0]
X_test, submission = datasets[1]
X_valid, y_valid = datasets[2]

nb_classes = y_train.shape[1]
print(nb_classes, 'classes')

dims = X_train.shape[1]
print(dims, 'dims')

model = Sequential()

model.add(Dense(1024, input_shape=(dims,)))
model.add(Dropout(0.1))#    input dropout
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(360))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(420))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
#opt=SGD(momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer="sgd")
auc_scores=[]
best_score=-1
best_model=None
threshold=0.9638
print('Training model...')
if need_validataion:
    for i in range(nb_epoch):
    #early_stopping=EarlyStopping(monitor='val_loss', patience=0, verbose=1)
    #model.fit(X_train, y_train, nb_epoch=nb_epoch,batch_size=256,validation_split=0.01,callbacks=[early_stopping])
        print ("best_score is:",best_score)
        model.fit(X_train, y_train, nb_epoch=1,batch_size=256)
        y_pre = model.predict_proba(X_valid)
        scores = roc_auc_score(y_valid,y_pre)
        auc_scores.append(scores)
        print (i,scores)
        if scores>best_score:
            best_score=scores
            best_model=model
            if best_score>threshold:
                y_pre = model.predict_proba(X_test)[:,1]
                save2model(submission,'keras_nn_test_'+str(best_score)+".csv", y_pre)
    plt.plot(auc_scores)
    plt.show()
else:
    model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=256)

if need_validataion:
    model=best_model
#print('Generating submission...')
y_pre = model.predict_proba(X_test)[:,1]
#print roc_auc_score(y_test,y_pre)
save2model(submission, 'keras_nn_test.csv',y_pre)
