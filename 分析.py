# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:03:59 2018

@author: ManSsssuper
"""
import pandas as pd

train_op=pd.read_csv(r"D:\DA_competition\DC\data\operation_train.csv")
train_tag=pd.read_csv(r"D:\DA_competition\DC\data\tag_train.csv")
train_tr=pd.read_csv(r"D:\DA_competition\DC\data\transaction_train.csv")
test_op=pd.read_csv(r"D:\DA_competition\DC\data\operation_test.csv")
test_tr=pd.read_csv(r"D:\DA_competition\DC\data\transaction_test.csv")
train_op=train_op.merge(train_tag,how="left",on="UID")
train_tr=train_tr.merge(train_tag,how="left",on="UID")
#处理时间
train_op["hour"]=train_op.time.apply(lambda x:x.split(":")[0])
train_tr["hour"]=train_tr.time.apply(lambda x:x.split(":")[0])
test_op["hour"]=test_op.time.apply(lambda x:x.split(":")[0])
test_tr["hour"]=test_tr.time.apply(lambda x:x.split(":")[0])

test_fts=pd.read_csv(r"D:\DA_competition\DC\data\sub_sample.csv")
test_fts=test_fts.drop("Tag",axis=1)
train_fts=train_tag.drop("Tag",axis=1)
know_UID=[]
know_UID.extend(train_op[train_op["version"]=="4.1.7"].UID.unique())
know_UID.extend(train_op[train_op["ip1"]=="0fe293bea342665a"].UID.unique())
know_UID.extend(train_tr[train_tr["channel"]==118].UID.unique())

know_UID.extend(train_tr[train_tr["amt_src1"]=="fd4d2d1006a95637"].UID.unique())

know_UID.extend(train_tr[train_tr["merchant"].isin(["8b3f74a1391b5427",
                                 "922720f3827ccef8","0e90f47392008def","5776870b5747e14e",
                                 "1f72814f76a984fa","2b2e7046145d9517","2260d61b622795fb",
                                 "6d55ccc689b910ee","4bca6018239c6201"])].UID.unique())
know_UID.extend(train_tr[train_tr["code1"]=="f1fa4af14fd5b68f"].UID.unique())
know_UID=pd.Series(know_UID).unique()
un_train_op=train_op[~train_op.UID.isin(know_UID)]
un_train_tr=train_tr[~train_tr.UID.isin(know_UID)]