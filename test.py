# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:20:35 2018

@author: ManSsssuper
"""

import pandas as pd
import numpy as np
s=pd.Series([1,2,3,4,2,2,2,3,4])
s1=pd.Series([2,2,3,4,1,2,3,4,np.nan])
s2=pd.Series([3,5,67,232,43,76,1,8,5])
s3=pd.Series([1,2,3,4,2,2,2,3,4])
df=pd.concat([s,s1,s2,s3],axis=1)
print(df)
df.columns=["s1","s2","s3","s4"]
x=df.groupby("s1")[["s2"]].agg(["nunique","count"]).reset_index(drop=False)
x.columns=["s1","nn","count"]
print(x)
print(df["s2"].mean())
21/8