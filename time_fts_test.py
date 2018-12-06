
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
train_op=pd.merge(train_op,train_tag,on="UID",how="left")
train_tr=pd.merge(train_tr,train_tag,on="UID",how="left")
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
        if field not in ["UID","time","Tag"]:
            op_fields.append(field)
    #指定tr提取特征的列
    for field in tr.columns:
        if field not in ["UID","time","Tag"]:
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

train=train.drop(["day_lencoder_op","success_lencoder_op",
                  "ip2_lencoder_op","ip2_sub_lencoder_op","market_code_lencoder_tr"],axis=1)
test=test.drop(["day_lencoder_op","success_lencoder_op",
                "ip2_lencoder_op","ip2_sub_lencoder_op","market_code_lencoder_tr"],axis=1)
print(train.shape)
    
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
#['mac2,f,count,op', 'wifi,f,nunique,op', 'wifi,f,count,op', 'ip2,f,count,op', 'ip2_sub,f,nunique,op', 'ip1_sub,f,nunique,tr']
#op字段
op_fields2=["mac1","mac2","wifi","device_code1","device_code2","device_code3","ip1","ip2","ip1_sub","ip2_sub"]
train=get_fields_as_key_fts3(train_op,op_fields2,train,"op")
test=get_fields_as_key_fts3(test_op,op_fields2,test,"op")

#tr字段
tr_fields2=["ip1","ip1_sub","acc_id1","acc_id2","acc_id3","mac1","device_code1","device_code2","device_code3"]
train=get_fields_as_key_fts3(train_tr,tr_fields2,train,"tr")
test=get_fields_as_key_fts3(test_tr,tr_fields2,test,"tr")

pre=["mac2,f,count,op", "wifi,f,nunique,op", "wifi,f,count,op","ip2,f,count,op", "ip2_sub,f,nunique,op", "ip1_sub,f,nunique,tr"]
after=["mean","max","min","sum"]

train=train.drop([x+","+y for x in pre for y in after],axis=1)
test=test.drop([x+","+y for x in pre for y in after],axis=1)

print(train.shape)

###################################################################################


day_and_time_tr_train=train_tr["day"].apply(str)+train_tr["time"]
train_tr["day_and_time"]=day_and_time_tr_train.apply(lambda x:int(x.replace(":","")))
day_and_time_op_train=train_op["day"].apply(str)+train_op["time"]
train_op["day_and_time"]=day_and_time_op_train.apply(lambda x:int(x.replace(":","")))
day_and_time_tr_test=test_tr["day"].apply(str)+test_tr["time"]
test_tr["day_and_time"]=day_and_time_tr_test.apply(lambda x:int(x.replace(":","")))
day_and_time_op_test=test_op["day"].apply(str)+test_op["time"]
test_op["day_and_time"]=day_and_time_op_test.apply(lambda x:int(x.replace(":","")))

#第一次转账金额
train_money_first=train_tr.groupby("UID").apply(lambda x:x[x.day_and_time==x.day_and_time.min()].trans_amt.agg(["max","min","sum","mean"]))
test_money_first=test_tr.groupby("UID").apply(lambda x:x[x.day_and_time==x.day_and_time.min()].trans_amt.agg(["max","min","sum","mean"]))
train_money_first.columns=["money_first_max","money_first_min","money_first_sum","money_first_mean"]
test_money_first.columns=["money_first_max","money_first_min","money_first_sum","money_first_mean"]
train_money_first=train_money_first.reset_index(drop=False)
test_money_first=test_money_first.reset_index(drop=False)
train=train.merge(train_money_first,how="left",on="UID")
test=test.merge(test_money_first,how="left",on="UID")
print(train.shape)
#最后一次转账金额
train_money_last=train_tr.groupby("UID").apply(lambda x:x[x.day_and_time==x.day_and_time.max()].trans_amt.agg(["max","min","sum","mean"]))
test_money_last=test_tr.groupby("UID").apply(lambda x:x[x.day_and_time==x.day_and_time.max()].trans_amt.agg(["max","min","sum","mean"]))
train_money_last.columns=["money_last_max","money_last_min","money_last_sum","money_last_mean"]
test_money_last.columns=["money_last_max","money_last_min","money_last_sum","money_last_mean"]
train_money_last=train_money_last.reset_index(drop=False)
test_money_last=test_money_last.reset_index(drop=False)
train=train.merge(train_money_last,how="left",on="UID")
test=test.merge(test_money_last,how="left",on="UID")
print(train.shape)
#发生第一次交易之前的操作记录条数
train_tr_first=pd.DataFrame(train_tr.groupby("UID").day_and_time.min())
train_tr_first.columns=["tr_first_day_and_time"]
train_tr_first["UID"]=train_tr_first.index
train_op=train_op.merge(train_tr_first,how="left",on="UID")
train_tr_first_op_count=pd.DataFrame(train_op[train_op.day_and_time>=train_op.tr_first_day_and_time].groupby("UID")["day_and_time"].count())
train_tr_first_op_count.columns=["op_before_tr_count"]
train_tr_first_op_count["UID"]=train_tr_first_op_count.index
train=train.merge(train_tr_first_op_count,how="left",on="UID")

test_tr_first=pd.DataFrame(test_tr.groupby("UID").day_and_time.min())
test_tr_first.columns=["tr_first_day_and_time"]
test_tr_first["UID"]=test_tr_first.index
test_op=test_op.merge(test_tr_first,how="left",on="UID")
test_tr_first_op_count=pd.DataFrame(test_op[test_op.day_and_time>=test_op.tr_first_day_and_time].groupby("UID")["day_and_time"].count())
test_tr_first_op_count.columns=["op_before_tr_count"]
test_tr_first_op_count["UID"]=test_tr_first_op_count.index
test=test.merge(test_tr_first_op_count,how="left",on="UID")
print(train.shape)
#发生最后一次交易之后的操作记录条数
train_tr_last=pd.DataFrame(train_tr.groupby("UID").day_and_time.max())
train_tr_last.columns=["tr_last_day_and_time"]
train_tr_last["UID"]=train_tr_last.index
train_op=train_op.merge(train_tr_last,how="left",on="UID")
train_tr_last_op_count=pd.DataFrame(train_op[train_op.day_and_time>=train_op.tr_last_day_and_time].groupby("UID")["day_and_time"].count())
train_tr_last_op_count.columns=["op_after_tr_count"]
train_tr_last_op_count["UID"]=train_tr_last_op_count.index
train=train.merge(train_tr_last_op_count,how="left",on="UID")

test_tr_last=pd.DataFrame(test_tr.groupby("UID").day_and_time.max())
test_tr_last.columns=["tr_last_day_and_time"]
test_tr_last["UID"]=test_tr_last.index
test_op=test_op.merge(test_tr_last,how="left",on="UID")
test_tr_last_op_count=pd.DataFrame(test_op[test_op.day_and_time>=test_op.tr_last_day_and_time].groupby("UID")["day_and_time"].count())
test_tr_last_op_count.columns=["op_after_tr_count"]
test_tr_last_op_count["UID"]=test_tr_last_op_count.index
test=test.merge(test_tr_last_op_count,how="left",on="UID")
print(train.shape)


#是否连着两天具有操作
def if_series(col):
    col_sort=col.sort_values().unique()
    max_abs=31
    if len(col_sort)==0:
        return -1
    for i in range(0,len(col_sort)-1):
        div=col_sort[i+1]-col_sort[i]
        if div<max_abs:
            max_abs=div
            if max_abs==1:
                return 1
    if max_abs==31:
        return -2
    return 0
def get_min_abs(col):
    col_sort=list(col.sort_values())
    max_abs=31
    if len(col_sort)==0:
        return -1
    for i in range(0,len(col_sort)-1):
        div=col_sort[i+1]-col_sort[i]
        if div<max_abs:
            max_abs=div
            if max_abs==0:
                return 0
    if max_abs==31:
        return -2
    return max_abs
def get_time_fea(data,df_by_uid,name):
    #是否连着两天有交易
    #是否连着两天有操作
    #第一次操作和最后一次操作间隔天数
    #第一次交易和最后一次交易间隔天数
    #两次交易的最短间隔
    #两次操作的最短间隔
    #用户交易日期方差
    #用户操作日期方差
    fts=data.groupby("UID").day.agg(["min","max",if_series,get_min_abs,"std"])
    fts.columns=[name+"_first_day_time_fea",name+"_last_day_time_fea",name+"_if_series_time_fea",
                 name+"_min_interval_time_fea",name+"_std_time_fea"]
    fts["UID"]=fts.index
    fts[name+"_max_interval_time_fea"]=fts[name+"_last_day_time_fea"]-fts[name+"_first_day_time_fea"]
    print(fts.columns)
    df_by_uid=df_by_uid.merge(fts,how="left",on="UID")
    
    return df_by_uid
train=get_time_fea(train_op,train,"op")
test=get_time_fea(test_op,test,"op")
train=get_time_fea(train_tr,train,"tr")
test=get_time_fea(test_tr,test,"tr")
print(train.shape)
#删除['op_first_day_time_fea', 'op_min_interval_time_fea', 'money_last_sum']
train=train.drop(['op_first_day_time_fea', 'op_min_interval_time_fea', 'money_last_sum'],axis=1)
test=test.drop(['op_first_day_time_fea', 'op_min_interval_time_fea', 'money_last_sum'],axis=1)
#第一次记录的labelencoder(op,tr)
#最后一次记录的labelencoder(op,tr)

#操作表和交易表之间相同字段的联系
###################################################################################
drop_cols=[]
for tag in ['op,min', 'tr,sum', 'op_sum']:
    for col in train.columns:
        if tag in col:
            drop_cols.append(col)
drop_cols.append('version_count')
train=train.drop(drop_cols,axis=1)
test=test.drop(drop_cols,axis=1)
#得到训练集和测试集顺便补充缺失值-1
train=train.fillna(-1)
test=test.fillna(-1)
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
#离散值查看
def get_rule_submit(data_train,data_test,train_min,test_min,ratio_1_min,ratio_0_max,fields,submit):
    for field in fields:
        g=data_train[["UID",field,"Tag"]].drop_duplicates().groupby(field).Tag
        train_1=g.apply(lambda x:(x==1).sum()/len(x))
        train_1=train_1[(train_1>=ratio_1_min)|(train_1<=ratio_0_max)]
        train_3=g.count()
        train_3=train_3[train_3>=train_min]
        data=pd.DataFrame(train_1).merge(pd.DataFrame(train_3),how="inner",left_index=True,right_index=True)
        
        num1=[]
        for ind in list(data.index):
            ind_2=data_test[data_test[field]==ind][["UID",field]].drop_duplicates().shape[0]
            num1.append(ind_2)
        data["test_people"]=num1
        data.columns=["ratio","train_all","test_people"]
        data=data[data.test_people>=test_min]
        f_value_1=data[data.ratio>=ratio_1_min]
        f_value_0=data[data.ratio<=ratio_0_max]
        
        s=np.array(submit["Tag"])
        if f_value_0.shape[0]>0:
            print(f_value_0)
            s[submit[submit.UID.isin(data_test[data_test[field].isin(list(f_value_0.index))].UID.unique())].index]=0
        if f_value_1.shape[0]>0:
            s[submit[submit.UID.isin(data_test[data_test[field].isin(list(f_value_1.index))].UID.unique())].index]=1
            print(f_value_1)
        submit.Tag=s
    return submit

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
    print(col,scores)
    
    submit=pd.concat([test_id,pd.Series(submit_preds)],axis=1,ignore_index=True)
    submit.columns=["UID","Tag"]
#    submit=get_rule_submit(train_op,test_op,100,100,1,0,test_op.columns[1:],submit)
#    submit=get_rule_submit(train_tr,test_tr,100,100,1,0,test_tr.columns[1:],submit)
    submit.to_csv(r"D:\DA_competition\DC\result\submit_%s.csv"%str(score),index=False)
    f=open(r"D:\Desktop\比赛\甜橙\fts_final_test.txt",mode='a')
    f.write(col+":"+str(scores)+"\n")
    f.close()
    return score

max_score=five(train,test,"all")
dels=[]
#for tag in ["f,nunique,op","f,nunique,tr","f,count,op","f,count,tr","tr,mean","op,mean","tr,max","op,max","tr,min","op,min","tr,sum","op_sum"]:
#    tag_drop=[]
#    for col in train.columns:
#        if tag in col:
#            tag_drop.append(col)
#    print(tag_drop)
#    train_tag=train.drop(tag_drop,axis=1)
#    test_tag=test.drop(tag_drop,axis=1)
#    tag_score=five(train_tag,test_tag,tag)
#    if tag_score>=max_score:
#        max_score=tag_score
#        train=train.drop(tag_drop,axis=1)
#        test=test.drop(tag_drop,axis=1)
#        dels.append(tag)
#f=open(r"D:\Desktop\比赛\甜橙\fts_final_test.txt",mode='a')
#f.write(str(dels)+"\n")
#f.close()
for col in train.columns:
    train_col=train.drop(col,axis=1)
    test_col=test.drop(col,axis=1)
    col_score=five(train_col,test_col,col)
    if col_score>=max_score:
        max_score=col_score
        train=train.drop(col,axis=1)
        test=test.drop(col,axis=1)
        dels.append(col)
f=open(r"D:\Desktop\比赛\甜橙\fts_final_test.txt",mode='a')
f.write(str(dels)+"\n")
f.close()
#dels=[]
#for col in train.columns:
#    if "time_fea" in col:
#        train_col=train.drop(col,axis=1)
#        test_col=test.drop(col,axis=1)
#        col_score=five(train_col,test_col,col)
#        if col_score>=max_score:
#            max_score=col_score
#            train=train.drop(col,axis=1)
#            test=test.drop(col,axis=1)
#            dels.append(col)
#
#
#print(dels)
#for col in ["op_before_tr_count","op_after_tr_count","money_first_max","money_first_min","money_first_sum","money_first_mean","money_last_max","money_last_min","money_last_sum","money_last_mean"]:
#    train_col=train.drop(col,axis=1)
#    test_col=test.drop(col,axis=1)
#    col_score=five(train_col,test_col,col)
#    if col_score>=max_score:
#        max_score=col_score
#        train=train.drop(col,axis=1)
#        test=test.drop(col,axis=1)
#        dels.append(col)
#print(dels)
