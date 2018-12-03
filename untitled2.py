import pandas as pd

train_op=pd.read_csv(r"D:\DA_competition\DC\data\operation_train.csv")
train_tag=pd.read_csv(r"D:\DA_competition\DC\data\tag_train.csv")
train_tr=pd.read_csv(r"D:\DA_competition\DC\data\transaction_train.csv")
test_op=pd.read_csv(r"D:\DA_competition\DC\data\operation_test.csv")
test_tr=pd.read_csv(r"D:\DA_competition\DC\data\transaction_test.csv")
train_op=pd.merge(train_op,train_tag,on="UID",how="left")
train_tr=pd.merge(train_tr,train_tag,on="UID",how="left")
#处理时间
train_op["hour"]=train_op.time.apply(lambda x:x.split(":")[0])
train_tr["hour"]=train_tr.time.apply(lambda x:x.split(":")[0])
test_op["hour"]=test_op.time.apply(lambda x:x.split(":")[0])
test_tr["hour"]=test_tr.time.apply(lambda x:x.split(":")[0])

test_fts=pd.read_csv(r"D:\DA_competition\DC\data\sub_sample.csv")
test_fts=test_fts.drop("Tag",axis=1)
train_fts=train_tag.drop("Tag",axis=1)
#离散值查看
def get_ratio(data_train,data_test,field):
    g=data_train[["UID",field,"Tag"]].drop_duplicates().groupby(field).Tag
    train_1=g.apply(lambda x:len(x[x==1])/len(x)).sort_values(ascending=False)
    train_1=train_1[(train_1>=0.95)|(train_1==0)]
    train_2=g.apply(lambda x:len(x[x==1]))
    
    train_3=g.count()
    train_3=train_3[train_3>=10]
    data=pd.DataFrame(train_1).merge(pd.DataFrame(train_3),how="inner",left_index=True,right_index=True)
    data=data.merge(pd.DataFrame(train_2),how="left",left_index=True,right_index=True)
    
    num1=[]
    for ind in list(data.index):
        ind_2=data_test[data_test[field]==ind][["UID",field]].drop_duplicates().shape[0]
        num1.append(ind_2)
    data["test_people"]=num1
    data.columns=["ratio","train_all","train_1","test_people"]
    data=data[data.test_people>0]
    return data