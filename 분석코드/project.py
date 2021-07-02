import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.sparse.linalg.eigen.arpack.arpack import svds
import os

prod = pd.read_csv("/home/joeun/product.csv")      
data = pd.read_csv("/home/joeun/orderlist.csv")   

# data = order.drop(["bnum","pname","price","quan"],axis=1)
data.insert(2, "paid", 1)
data['uid']=data['member'].factorize()[0]
#순서대로 고유값 부여
# print(data)
pv = data.pivot_table("paid",index="uid", columns="pnum")
pv_data = pv.fillna(0)
df_pv_data = DataFrame(pv_data)
# print(df_pv_data)
matrix = df_pv_data.values
paid_mean = np.mean(matrix,axis=1)
matrix_mean = matrix - paid_mean.reshape(-1,1)
# print(matrix_mean)
U , sigma, Vt = svds(matrix_mean, k=12)
sigma = np.diag(sigma)
svd_data= np.dot(np.dot(U,sigma),Vt) + paid_mean.reshape(-1,1)
df_svd_preds = DataFrame(svd_data,columns = df_pv_data.columns)
# print(df_svd_preds)

def recommend(df_svd_preds, user_id, ori_prod, ori_data, num_recommendations=5):
    user_row_number = user_id
    sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)
    user_data = ori_data[ori_data.uid == user_id]
    user_history = user_data.merge(ori_prod, on = 'pnum').sort_values(["paid"], ascending=False)
    recommendations = ori_prod[~ori_prod['pnum'].isin(user_history['pnum'])]
    recommendations = recommendations.merge( pd.DataFrame(sorted_user_predictions).reset_index(), on = 'pnum')
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]
    return user_history.iloc[:,[3,0]], recommendations.iloc[:,[0]] 

# already_paid, predictions = recommend(df_svd_preds, 4999, prod, data, 5)
# print(predictions)
# print(already_paid)
# result = pd.concat([already_paid.iloc[[0],[0,1]],predictions], ignore_index=True)
# result['uid']=result['uid'].fillna(0).astype(int)
# result['pnum']=result['pnum'].fillna(0).astype(int)
# result['uid'] = result['uid'][0]
# result['member'] = result['member'][0]
# result=result.drop(index=0,axis=0)
# print(result)

if os.path.exists("/home/joeun/result.txt"):
    os.remove("/home/joeun/result.txt")
else:
    pass
    
i=0
a = pd.Series.unique(data['uid'].astype(int))
for i in range(10):
    already_paid, predictions = recommend(df_svd_preds, i, prod, data, 5)
    result = pd.concat([already_paid,predictions], ignore_index=True)
    result['uid']=result['uid'].fillna(0).astype(int)
    result['pnum']=result['pnum'].fillna(0).astype(int)
    result['uid'] = result['uid'][0]
    result['member'] = result['member'][0]
    result=result[result.pnum != 0]
    print(result)
    result.to_csv('/home/joeun/result.txt',mode='a',header=False,index=False)
