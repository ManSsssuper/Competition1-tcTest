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
#删除缺失值，保留重复值分数更高
#test_op=test_op.drop_duplicates()
#test_tr=test_tr.drop_duplicates()

##处理时间
#train_op["hour"]=train_op.time.apply(lambda x:x.split(":")[0])
#train_tr["hour"]=train_tr.time.apply(lambda x:x.split(":")[0])
#test_op["hour"]=test_op.time.apply(lambda x:x.split(":")[0])
#test_tr["hour"]=test_tr.time.apply(lambda x:x.split(":")[0])

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
        if field not in ["UID","day"]:
            op_fields.append(field)
    #指定tr提取特征的列
    for field in tr.columns:
        if field not in ["UID","day"]:
            tr_fields.append(field)
    
    #提取nunique和count特征
    df_by_uid=df_by_uid.merge(op.groupby("UID")[op_fields].agg(["nunique","count"]).reset_index(drop=False),how="left",on="UID")
    df_by_uid=df_by_uid.merge(tr.groupby("UID")[tr_fields].agg(["nunique","count"]).reset_index(drop=False),how="left",on="UID")
    
    #提取transaction表中的money的特征
    df_by_uid=df_by_uid.merge(tr.groupby("UID")[money_fields].agg(["max","min","mean","std","sum"])
                                                        .reset_index(drop=False),how="left",on="UID")
    return df_by_uid

#得到训练集和测试集顺便补充缺失值-1
train=get_fea_base(train_op,train_tr,train_fts).fillna(-1)
test=get_fea_base(test_op,test_tr,test_fts).fillna(-1)

train=train.drop("UID",axis=1)
train_y=train_tag["Tag"]

test_id=test_fts["UID"]
test=test.drop("UID",axis=1)


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
    
    params={'boosting':'gbdt',
        'objective': 'binary',
        'learning_rate': 0.1,
        'max_depth': 6,
        'num_leaves': 31,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'min_child_weight': 25,
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
submit.to_csv(r"D:\DA_competition\DC\result\submit_%s.csv"%str(score),index=False)






    
    
            