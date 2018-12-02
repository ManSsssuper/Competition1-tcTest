
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:03:59 2018

@author: ManSsssuper
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

train_op=pd.read_csv(r"D:\DA_competition\DC\data\operation_train.csv")
train_tag=pd.read_csv(r"D:\DA_competition\DC\data\tag_train.csv")
train_tr=pd.read_csv(r"D:\DA_competition\DC\data\transaction_train.csv")
test_op=pd.read_csv(r"D:\DA_competition\DC\data\operation_test.csv")
test_tr=pd.read_csv(r"D:\DA_competition\DC\data\transaction_test.csv")
 
#处理时间
train_op["hour"]=train_op.time.apply(lambda x:x.split(":")[0])
train_tr["hour"]=train_tr.time.apply(lambda x:x.split(":")[0])
test_op["hour"]=test_op.time.apply(lambda x:x.split(":")[0])
test_tr["hour"]=test_tr.time.apply(lambda x:x.split(":")[0])

test_fts=pd.read_csv(r"D:\DA_competition\DC\data\sub_sample.csv")
test_fts=test_fts.drop("Tag",axis=1)
train_fts=train_tag.drop("Tag",axis=1)


#得到基础特征
#得到基础特征

def get_fea_base(op,tr,df_by_uid):
    op_fields=[]
    tr_fields=[]
    money_fields=["trans_amt","bal"]
    #指定op提取特征的列
    for field in op.columns:
        if field not in ["UID","time"]:
            op_fields.append(field)
    #指定tr提取特征的列
    for field in tr.columns:
        if field not in ["UID","time"]:
            tr_fields.append(field)
    #提取nunique和count特征
    op_fts=op.groupby("UID")[op_fields].agg(["nunique","count"])
    op_fts.columns=["_".join(x) for x in op_fts.columns.ravel()]
    op_fts=op_fts.reset_index(drop=False)
    df_by_uid=df_by_uid.merge(op_fts,how="left",on="UID")
    
    tr_fts1=tr.groupby("UID")[tr_fields].agg(["nunique","count"])
    tr_fts1.columns=["_".join(x) for x in tr_fts1.columns.ravel()]
    tr_fts1=tr_fts1.reset_index(drop=False)
    df_by_uid=df_by_uid.merge(tr_fts1,how="left",on="UID")
    
    #提取transaction表中的money的特征
    tr_fts2=tr.groupby("UID")[money_fields].agg(["max","min","mean","std","sum"])
    tr_fts2.columns=["_".join(x) for x in tr_fts2.columns.ravel()]
    tr_fts2=tr_fts2.reset_index(drop=False)
    df_by_uid=df_by_uid.merge(tr_fts2,how="left",on="UID")
    return df_by_uid


#得到训练集和测试集顺便补充缺失值-1
train=get_fea_base(train_op,train_tr,train_fts)
test=get_fea_base(test_op,test_tr,test_fts)

train=train.drop(['mode_count', 'os_count', 'device_code1_nunique_x', 
                  'ip1_count_x', 'ip2_count', 'hour_count_x', 'channel_count', 
                  'day_count_y', 'trans_amt_count', 'amt_src1_count', 'merchant_count',
                  'code2_nunique', 'trans_type1_count', 'device2_count_y', 'ip1_count_y',
                  'bal_count', 'amt_src2_nunique', 'market_code_count', 'trans_amt_std'],axis=1)
test=test.drop(['mode_count', 'os_count', 'device_code1_nunique_x', 
                  'ip1_count_x', 'ip2_count', 'hour_count_x', 'channel_count', 
                  'day_count_y', 'trans_amt_count', 'amt_src1_count', 'merchant_count',
                  'code2_nunique', 'trans_type1_count', 'device2_count_y', 'ip1_count_y',
                  'bal_count', 'amt_src2_nunique', 'market_code_count', 'trans_amt_std'],axis=1)
print(train.shape)

def get_label_in_coder_fts(train_data,test_data,fields,train,test,name,first=True):
    fields.append("UID")
    data=pd.concat([train_data[fields],test_data[fields]],axis=0,ignore_index=True)
    data=data.applymap(str)
    data=data.fillna("-1")
    if first:
        s=data.groupby("UID").agg(lambda x:list(pd.Series.mode(x))[0])
    else:
        s=data.groupby("UID").agg(lambda x:list(pd.Series.mode(x))[-1])
    
    fts=pd.DataFrame(s.index).applymap(int)
    le=LabelEncoder()
    for col in s.columns:
        le.fit(s[col])
        fts[col+"_lencoder_"+name]=le.transform(s[col])
    train=train.merge(fts,how='left',on='UID')
    test=test.merge(fts,how='left',on='UID')
    return train,test
train,test=get_label_in_coder_fts(train_op,test_op,list(test_op.columns[1:]),train,test,"op",False)
train,test=get_label_in_coder_fts(train_tr,test_tr,list(test_tr.columns[1:]),train,test,"tr",False)
print(train.shape)
train=train.drop(["day_lencoder_op","success_lencoder_op",
                  "ip2_lencoder_op","ip2_sub_lencoder_op","market_code_lencoder_tr"],axis=1)
test=test.drop(["day_lencoder_op","success_lencoder_op",
                "ip2_lencoder_op","ip2_sub_lencoder_op","market_code_lencoder_tr"],axis=1)

#################################添加测试特征#######################################
#添加上测试特征


#添加那几个关键字段的count
def get_fields_as_key_fts3(data,fields,df_by_uid,name):
    for field in fields:
        f_count=data.groupby(field)["UID"].agg(["nunique","count"])

        
        f_count.columns=[field+",f,nunique,"+name,field+",f,count,"+name]
        f_count=f_count.reset_index(drop=False)
        fts_mid=pd.merge(data[["UID",field]],f_count,how="left",on=field).drop_duplicates()
        fts_mid=fts_mid.drop(field,axis=1)
        
        fts=fts_mid.groupby("UID")[[field+",f,nunique,"+name,field+",f,count,"+name]].agg(["mean","max","min","sum"])
        fts.columns=[",".join(x) for x in fts.columns.ravel()]
        fts=fts.reset_index(drop=False)      
        df_by_uid=pd.merge(df_by_uid,fts,how="left",on="UID")
    return df_by_uid

#op字段
op_fields2=["mac1","mac2","wifi","device_code1","device_code2","device_code3","ip1","ip2","ip1_sub","ip2_sub"]
train=get_fields_as_key_fts3(train_op,op_fields2,train,"op")
test=get_fields_as_key_fts3(test_op,op_fields2,test,"op")
print(train.shape)

#tr字段
tr_fields2=["ip1","ip1_sub","acc_id1","acc_id2","acc_id3","mac1","device_code1","device_code2","device_code3"]
train=get_fields_as_key_fts3(train_tr,tr_fields2,train,"tr")
test=get_fields_as_key_fts3(test_tr,tr_fields2,test,"tr")
print(train.shape)
pre=["mac2,f,count,op", "wifi,f,nunique,op", "wifi,f,count,op","ip2,f,count,op", "ip2_sub,f,nunique,op", "ip1_sub,f,nunique,tr"]
after=["mean","max","min","sum"]

train=train.drop([x+","+y for x in pre for y in after],axis=1)
test=test.drop([x+","+y for x in pre for y in after],axis=1)
train=train.fillna(-1)
test=test.fillna(-1)

##################################################################################
def get_fts_1(data_train,data_test,fields,trainf,testf):
    data_one_hot=pd.get_dummies(pd.concat([data_train[fields].applymap(str),data_test[fields].applymap(str)],axis=0),dummy_na=True)
    train_data_one_hot=data_one_hot.iloc[:len(data_train),:].reset_index(drop=True)
    test_data_one_hot=data_one_hot.iloc[len(data_train):,:].reset_index(drop=True)
    
    train_data_one_hot.columns=["cross_type_"+col for col in train_data_one_hot.columns]
    train_data_one_hot["UID"]=data_train["UID"]
    
    test_data_one_hot.columns=["cross_type_"+col for col in test_data_one_hot.columns]
    test_data_one_hot["UID"]=data_test["UID"]
    trainf=trainf.merge(train_data_one_hot.groupby("UID").sum().reset_index(drop=False),how="left",on="UID")
    testf=testf.merge(test_data_one_hot.groupby("UID").sum().reset_index(drop=False),how="left",on="UID")
    return trainf,testf
op_cross_type=["success","os","version"]
train,test=get_fts_1(train_op,test_op,op_cross_type,train,test)
tr_cross_type=["channel","amt_src1","trans_type1","trans_type2"]
train,test=get_fts_1(train_tr,test_tr,tr_cross_type,train,test)

train_y=train_tag["Tag"]
test_id=test_fts["UID"]
######################################模型训练#######################################
def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

######################模型验证#######################################
"""
    尝试一下是预测5次除以5还是训练集全集预测一次提交效果哪个好
"""

def five(train,test,col):
    valid_preds=np.zeros(train.shape[0])
    submit_preds=np.zeros(test.shape[0])
    scores=[]
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    for index,(train_index,valid_index) in enumerate(skf.split(train,train_y)):
        train_set=lgb.Dataset(train.iloc[train_index],train_y.iloc[train_index])
        valid_X=train.iloc[valid_index]
        valid_y=train_y.iloc[valid_index]
        
        params={
            'boosting':'gbdt',
            'objective': 'binary',
            'learning_rate': 0.04,
            'max_depth': 8,
            'num_leaves':100,
            'lambda_l1': 0.1,
#            'subsample_by_tree':0.9,
            }
        n_rounds=1200
        clf=lgb.train(params,train_set,n_rounds)
        
        valid_pred=clf.predict(valid_X)
        scores.append(tpr_weight_funtion(valid_y,valid_pred))
        valid_preds[valid_index]=valid_pred
        
        #最终结果得到方式是预测五次求均值
        sub_pred=clf.predict(test)
        submit_preds+=sub_pred/5
        
    ######################生成提交结果##################################################
    score=tpr_weight_funtion(train_y,valid_preds)
    scores.append(score)
    print(scores)
    submit=pd.concat([test_id,pd.Series(submit_preds)],axis=1,ignore_index=True)
    submit.columns=["UID","Tag"]
    submit.to_csv(r"D:\DA_competition\DC\result\submit_%s.csv"%str(score),index=False)
    f=open(r"D:\Desktop\比赛\甜橙\f_count_One_hot_fts_test_在添加le基础上.txt",mode='a')
    f.write(col+":"+str(scores)+"\n")
    f.close()
    return score
max_score=five(train,test,"base:")
dels=[]
####################################################################################
for i in ["success","os","version","channel","amt_src1","trans_type1","trans_type2"]:
    ctname="cross_type_"+i
    i_fts=[]
    for c in train.columns:
        if ctname in c:
            i_fts.append(c)
    train_i=train.drop(i_fts,axis=1)
    test_i=test.drop(i_fts,axis=1)
    i_score=five(train_i,test_i,ctname)
    if i_score>=max_score:
        train=train.drop(i_fts,axis=1)
        test=test.drop(i_fts,axis=1)
        dels.append(ctname)
f=open(r"D:\Desktop\比赛\甜橙\f_count_One_hot_fts_test_在添加le基础上.txt",mode='a')
f.write(str(dels)+"\n")
f.close()



