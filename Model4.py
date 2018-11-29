# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:03:59 2018

@author: ManSsssuper
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

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
def get_fea_base(op,tr,df_by_uid):
    op_fields=[]
    tr_fields=[]
    money_fields=["trans_amt","bal"]
    #指定op提取特征的列
    for field in op.columns:
        if field not in ["UID","day","time"]:
            op_fields.append(field)
    #指定tr提取特征的列
    for field in tr.columns:
        if field not in ["UID","day","time"]:
            tr_fields.append(field)
    
    #提取nunique和count特征
    df_by_uid=df_by_uid.merge(op.groupby("UID")[op_fields].agg(["nunique","count"]).reset_index(drop=False),how="left",on="UID")
    df_by_uid=df_by_uid.merge(tr.groupby("UID")[tr_fields].agg(["nunique","count"]).reset_index(drop=False),how="left",on="UID")
    
    #提取transaction表中的money的特征
    df_by_uid=df_by_uid.merge(tr.groupby("UID")[money_fields].agg(["max","min","mean","std","sum"])
                                                        .reset_index(drop=False),how="left",on="UID")
    return df_by_uid


#得到训练集和测试集顺便补充缺失值-1
train=get_fea_base(train_op,train_tr,train_fts)
test=get_fea_base(test_op,test_tr,test_fts)
print(train.shape)

#################################添加测试特征#######################################
#添加上测试特征


def get_fts_1(data_train,data_test,fields,trainf,testf):
    data_one_hot=pd.get_dummies(pd.concat([data_train[fields].applymap(str),data_test[fields].applymap(str)],axis=0),dummy_na=True)
    train_data_one_hot=data_one_hot.iloc[:len(data_train),:].reset_index(drop=True)
    test_data_one_hot=data_one_hot.iloc[len(data_train):,:].reset_index(drop=True)
    train_data_one_hot["UID"]=data_train["UID"]
    test_data_one_hot["UID"]=data_test["UID"]

    trainf=trainf.merge(train_data_one_hot.groupby("UID").sum().reset_index(drop=False),how="left",on="UID")
    testf=testf.merge(test_data_one_hot.groupby("UID").sum().reset_index(drop=False),how="left",on="UID")
    return trainf,testf

#op离散字段的cross_type值,涨分
op_fields1=["mode","success","os","version"]
#op_fields1=["success","os","version"]
train,test=get_fts_1(train_op,test_op,op_fields1,train,test)
print(train.shape)

#tr离散字段的cross_type值,
tr_fields1=["channel","amt_src1","trans_type1","amt_src2","trans_type2"]
#tr_fields1=["channel","amt_src1","trans_type1","trans_type2"]
train,test=get_fts_1(train_tr,test_tr,tr_fields1,train,test)
print(train.shape)

#添加那几个关键字段的count
def get_fields_as_key_fts3(data,fields,df_by_uid):
    for field in fields:
        f_count=data.groupby(field)["UID"].agg(["nunique","count"]).reset_index(drop=False)
        f_count.columns=[field,field+"nunique",field+"count"]
        
        fts_mid=pd.merge(data[["UID",field]],f_count,how="left",on=field).drop_duplicates()
        fts_mid=fts_mid.drop(field,axis=1)
        
        fts=fts_mid.groupby("UID")[[field+"nunique",field+"count"]].agg(["mean","max","min","sum"])
        fts["UID"]=fts.index
        
        df_by_uid=pd.merge(df_by_uid,fts,how="left",on="UID")
    return df_by_uid

#op字段
op_fields2=["mac1","mac2","wifi","device_code1","device_code2","device_code3","ip1","ip2","ip1_sub","ip2_sub"]
train=get_fields_as_key_fts3(train_op,op_fields2,train)
test=get_fields_as_key_fts3(test_op,op_fields2,test)
print(train.shape)

#tr字段
tr_fields2=["ip1","ip1_sub","acc_id1","acc_id2","acc_id3","mac1","device_code1","device_code2","device_code3"]
train=get_fields_as_key_fts3(train_tr,tr_fields2,train)
test=get_fields_as_key_fts3(test_tr,tr_fields2,test)
print(train.shape)
##添加空值率
#def get_tr_null_fts6(data,df_by_uid):
#    fields=["device_code1","device_code2","device1","mac1","geo_code","ip1","ip1_sub","device2"]
#    group=data.groupby("UID")
#    fts=pd.DataFrame(group.apply(lambda x:len(x)))
#    fts.columns=["count"]
#    for field in fields:
#        fts=pd.concat([fts,group[field].apply(lambda x:len(x[x.isnull()]))],axis=1)
#    fts=fts.div(fts["count"],axis=0)
#    fts=fts.reset_index(drop=False)
#    fts=fts.drop(["count"],axis=1)
#    df_by_uid=pd.merge(df_by_uid,fts,how="left",on="UID")
#    return df_by_uid
#train=get_tr_null_fts6(train_tr,train)
#test=get_tr_null_fts6(test_tr,test)

##添加有操作无交易
#def get_no_tr_fts(tr,df_by_uid):
#    s=np.zeros(len(df_by_uid))
#    
#    s[df_by_uid[~df_by_uid["UID"].isin(tr["UID"])].index]=1
#    df_by_uid["no_tr"]=s
#    return df_by_uid
#
#train=get_no_tr_fts(train_tr,train)
#test=get_no_tr_fts(test_tr,test)
#
##有交易无操作的标志位
#def get_no_op_fts(op,df_by_uid):
#    s=np.zeros(len(df_by_uid))
#    s[df_by_uid[~df_by_uid["UID"].isin(op["UID"])].index]=1
#    df_by_uid["no_op"]=s
#    return df_by_uid
#
#train=get_no_op_fts(train_op,train)
#test=get_no_op_fts(test_op,test)

##根据trans_type分类统计金钱总额
#def get_type_money(data_train,data_test,trainf,testf):
#    data=pd.concat([data_train,data_test],axis=0)
#    
#    fea=data.groupby(["UID","trans_type1"])["trans_amt"].agg(["max","min","sum","mean","std"]).unstack()
#    fea["UID"]=fea.index
#    trainf=trainf.merge(fea,how="left",on="UID")
#    testf=testf.merge(fea,how="left",on="UID")
#    trainf=trainf.fillna(0)
#    testf=testf.fillna(0)
#
#    return trainf,testf
#train,test=get_type_money(train_tr,test_tr,train,test)

#是否是iphone

def get_op_fts_if_iphone(op,df_by_uid):
    
    device2=op.device2.reset_index(drop=True).apply(lambda x:str(x))
    isIphone=np.zeros(len(op))
    for i in range(0,len(op)):
        if ("IPHONE 5S" in device2[i])|("IPHONE 5" in device2[i])|("IPHONE 5C" in device2[i]):
            isIphone[i]=1
    op["device2_new"]=isIphone
    s=op.groupby('UID').device2_new.max().reset_index(drop=False)
    df_by_uid=df_by_uid.merge(s,how='left',on='UID')
    return df_by_uid
train=get_op_fts_if_iphone(train_op,train)
test=get_op_fts_if_iphone(test_op,test)
#发生第一次转账之前的操作记录条数

#发生最后一次转账之后的操作记录条数

#用户有交易无操作的天数、操作种类、操作数
#用户有操作无交易的天数、操作种类、操作数
#用户有操作有交易当天的分析

train=train.fillna(-1)
test=test.fillna(-1)

##################################################################################
#train=train.drop("UID",axis=1)
train_y=train_tag["Tag"]

test_id=test_fts["UID"]
#test=test.drop("UID",axis=1)


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
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 50,
        'lambda_l1': 0.1,
        
        'subsample_by_tree':0.7,
        }
    n_rounds=1000
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
submit.to_csv(r"D:\DA_competition\DC\result\submit_添加type金钱_%s.csv"%str(score),index=False)






    
    
            