# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:41:32 2018

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
train_base=get_fea_base(train_op,train_tr,train_fts)
test_base=get_fea_base(test_op,test_tr,test_fts)
print(train_base.shape)

#################################添加测试特征#######################################
#添加上测试特征

train_base=train_base.fillna(-1)
test_base=test_base.fillna(-1)

##################################################################################
#train=train.drop("UID",axis=1)
train_y=train_tag["Tag"]

test_id=test_fts["UID"]
#test=test.drop("UID",axis=1)
train_base=train_base.drop(['mode_count', 'os_count', 'device_code1_nunique_x', 
                  'ip1_count_x', 'ip2_count', 'hour_count_x', 'channel_count', 
                  'day_count_y', 'trans_amt_count', 'amt_src1_count', 'merchant_count',
                  'code2_nunique', 'trans_type1_count', 'device2_count_y', 'ip1_count_y',
                  'bal_count', 'amt_src2_nunique', 'market_code_count', 'trans_amt_std'],axis=1)
test_base=test_base.drop(['mode_count', 'os_count', 'device_code1_nunique_x', 
                  'ip1_count_x', 'ip2_count', 'hour_count_x', 'channel_count', 
                  'day_count_y', 'trans_amt_count', 'amt_src1_count', 'merchant_count',
                  'code2_nunique', 'trans_type1_count', 'device2_count_y', 'ip1_count_y',
                  'bal_count', 'amt_src2_nunique', 'market_code_count', 'trans_amt_std'],axis=1)
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
def get_score(train,test,col):
    valid_preds=np.zeros(train.shape[0])
    submit_preds=np.zeros(test.shape[0])
    scores=[]
    n=5
    skf=StratifiedKFold(n_splits=n,random_state=0,shuffle=True)
    for index,(train_index,valid_index) in enumerate(skf.split(train,train_y)):
        train_set=lgb.Dataset(train.iloc[train_index],train_y.iloc[train_index])
        valid_X=train.iloc[valid_index]
        valid_y=train_y.iloc[valid_index]
        
        params={
            'boosting':'gbdt',
            'objective': 'binary',
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves':100,
            'lambda_l1': 0.1,
#            'subsample_by_tree':0.9,
            'silent':True
            }
        n_rounds=1100
        clf=lgb.train(params,train_set,n_rounds)
        
        valid_pred=clf.predict(valid_X)
        scores.append(tpr_weight_funtion(valid_y,valid_pred))
        valid_preds[valid_index]=valid_pred
        
        #最终结果得到方式是预测五次求均值
        sub_pred=clf.predict(test)
        submit_preds+=sub_pred/n
        
    ######################生成提交结果##################################################
    score=tpr_weight_funtion(train_y,valid_preds)
    score_mean=sum(scores)/n
    scores.append(score_mean)
    scores.append(score)
    print(col,scores)
    f=open(r"D:\Desktop\比赛\甜橙\base_fts_test.txt",mode='a')
    f.write(col+":"+str(scores)+"\n")
    f.close()
    return score
max_score=get_score(train_base,test_base,"base:")
dels=[]
for col in train_base.columns:
    train=train_base.drop(col,axis=1)
    test=test_base.drop(col,axis=1)
    """
    尝试一下是预测5次除以5还是训练集全集预测一次提交效果哪个好
    """
    col_score=get_score(train,test,col)
    if col_score>=max_score:
        max_score=col_score
        train_base=train_base.drop(col,axis=1)
        test_base=test_base.drop(col,axis=1)
        print("删除：",col)
        dels.append(col)
    else:
        print("保留：",col)
	
print(dels)
        






    
    
            
