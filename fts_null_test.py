
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

##################################################################################

def get_null_fts(op,tr,df_by_uid):
    op_null=pd.DataFrame((op.isnull()).sum(axis=1))
    op_null.columns=["null_count"]
    
    tr_null=pd.DataFrame((tr.isnull()).sum(axis=1))
    tr_null.columns=["null_count"]
    
    op_null["UID"]=op["UID"]
    tr_null["UID"]=tr["UID"]

#    op_fts=op_null.groupby("UID")["null_count"].agg(["max","min","mean","sum","std"])
#    op_fts.columns=["op_null_count_max","op_null_count_min","op_null_count_mean","op_null_count_sum","op_null_count_std"]
#    op_fts=op_fts.reset_index(drop=False)
#    
#    tr_fts=tr_null.groupby("UID")["null_count"].agg(["max","min","mean","sum","std"])
#    tr_fts.columns=["tr_null_count_max","tr_null_count_min","tr_null_count_mean","tr_null_count_sum","tr_null_count_std"]
#    tr_fts=tr_fts.reset_index(drop=False)
#    
#    df_by_uid=df_by_uid.merge(op_fts,how="left",on="UID")
#    df_by_uid=df_by_uid.merge(tr_fts,how="left",on="UID")
    
    all_null=pd.concat([op_null,tr_null],axis=0)
    all_fts=all_null.groupby("UID")["null_count"].agg(["max","min","mean","sum","std"])
    all_fts.columns=["all_null_count_max","all_null_count_min","all_null_count_mean","all_null_count_sum","all_null_count_std"]
    all_fts=all_fts.reset_index(drop=False)
    df_by_uid=df_by_uid.merge(all_fts,how="left",on="UID")
    
    return df_by_uid
train=get_null_fts(train_op,train_tr,train)
test=get_null_fts(test_op,test_tr,test)
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
##离散值查看
#def get_rule_submit(data_train,data_test,train_min,test_min,ratio_1_min,ratio_0_max,fields,submit):
#    for field in fields:
#        g=data_train[["UID",field,"Tag"]].drop_duplicates().groupby(field).Tag
#        train_1=g.apply(lambda x:(x==1).sum()/len(x))
#        train_1=train_1[(train_1>=ratio_1_min)|(train_1<=ratio_0_max)]
#        train_3=g.count()
#        train_3=train_3[train_3>=train_min]
#        data=pd.DataFrame(train_1).merge(pd.DataFrame(train_3),how="inner",left_index=True,right_index=True)
#        
#        num1=[]
#        for ind in list(data.index):
#            ind_2=data_test[data_test[field]==ind][["UID",field]].drop_duplicates().shape[0]
#            num1.append(ind_2)
#        data["test_people"]=num1
#        data.columns=["ratio","train_all","test_people"]
#        data=data[data.test_people>=test_min]
#        f_value_1=data[data.ratio>=ratio_1_min]
#        f_value_0=data[data.ratio<=ratio_0_max]
#        
#        s=np.array(submit["Tag"])
#        if f_value_0.shape[0]>0:
#            s[submit[submit.UID.isin(data_test[data_test[field].isin(list(f_value_0.index))].UID.unique())].index]=0
#        if f_value_1.shape[0]>0:
#            s[submit[submit.UID.isin(data_test[data_test[field].isin(list(f_value_1.index))].UID.unique())].index]=1
#        submit.Tag=s
#    return submit
def get_result(train,test,col):

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
    return score
max_score=get_result(train,test,"base")
dels=[]
for col in train.columns:
    if "null_count" in col:
        train_col=train.drop(col,axis=1)
        test_col=test.drop(col,axis=1)
        col_score=get_result(train_col,test_col,col)
        if col_score>=max_score:
            max_score=col_score
            train=train.drop(col,axis=1)
            test=test.drop(col,axis=1)
            dels.append(col)
print(dels)
#base [0.7998833138856476, 0.78389731621936987, 0.76639439906651119, 0.76312718786464406, 0.77794632438739786, 0.78170361726954496]
#op_null_count_max [0.80781796966161024, 0.78728121353558922, 0.76301050175029173, 0.76662777129521575, 0.7796966161026837, 0.77885647607934649]
#op_null_count_min [0.798833138856476, 0.78086347724620764, 0.76207701283547258, 0.76371061843640609, 0.77782963827304552, 0.77864644107351211]
#op_null_count_mean [0.81003500583430577, 0.78891481913652273, 0.75962660443407248, 0.76324387397899651, 0.77701283547257882, 0.78123687281213527]
#op_null_count_sum [0.80443407234539088, 0.78436406067677955, 0.76382730455075842, 0.76336056009334885, 0.78004667444574094, 0.78231038506417738]
#op_null_count_std [0.8007001166861144, 0.78564760793465582, 0.76301050175029173, 0.7591598599766628, 0.77771295215869318, 0.77988331388564758]
#tr_null_count_max [0.798833138856476, 0.78366394399066519, 0.76382730455075842, 0.75974329054842471, 0.78121353558926487, 0.78245040840140034]
#tr_null_count_min [0.80466744457409567, 0.78226371061843647, 0.761610268378063, 0.76219369894982503, 0.78098016336056009, 0.77962660443407239]
#tr_null_count_mean [0.80023337222870483, 0.78238039673278881, 0.76814469078179692, 0.76522753792298714, 0.7796966161026837, 0.77985997666277707]
#tr_null_count_sum [0.80630105017502918, 0.78844807467911326, 0.76674445740956809, 0.76907817969661607, 0.77654609101516914, 0.77990665110851809]
#tr_null_count_std [0.80560093348891482, 0.78203033838973157, 0.76931155192532086, 0.76056009334889152, 0.77794632438739797, 0.78219369894982504]
#['op_null_count_sum', 'tr_null_count_max']
#['op_null_count_sum', 'tr_null_count_max']
#base [0.80595099183197194, 0.78343057176196029, 0.76324387397899662, 0.76254375729288215, 0.77829638273045509, 0.78063010501750285]
#all_null_count_max [0.80525087514585758, 0.78249708284714115, 0.76207701283547258, 0.75694282380396738, 0.77782963827304541, 0.77654609101516914]
#all_null_count_min [0.8015169194865811, 0.78343057176196029, 0.76476079346557757, 0.76207701283547258, 0.77934655775962658, 0.77960326721120188]
#all_null_count_mean [0.80058343057176207, 0.79113185530921815, 0.75904317386231046, 0.76207701283547258, 0.77946324387397903, 0.78051341890315051]
#all_null_count_sum [0.80385064177362897, 0.78296382730455083, 0.75950991831972003, 0.75974329054842471, 0.77841306884480743, 0.78144690781796966]
#all_null_count_std [0.80280046674445737, 0.78308051341890317, 0.76849474912485416, 0.76067677946324386, 0.77899649941656945, 0.77971995332555433]
#['all_null_count_sum']