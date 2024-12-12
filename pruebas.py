import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.pipeline import Pipeline
import datetime
import pandas as pd

print(0 ==None)


# date = pd.to_datetime(1000000000, unit='s')
# # date = datetime.datetime.fromtimestamp(1000000000)
# tsdiff = (date + pd.DateOffset(months=10))-date
# print(tsdiff.as_unit('days'))

# with open("bla.zlib", "wb") as we:
#     joblib.dump(Pipeline([RandomForestClassifier()]),we)

# with open("bla.zlib", "rb") as we:
#     print(joblib.load(we))

# print(RandomForestClassifier().__class__.__name__)
# a = np.array([[1,2,3],[1,2,3]])
# b = a + np.array([[1],[3]])
# print(b==4)
# print((b==4).astype(int))