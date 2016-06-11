import blaze as blz
import numpy as np
import csv as csv
import math

print("preprocessing...")

# Data cleanup
# TRAIN DATA
train_df = blz.data('data/train.csv')        # Load the train file into a dataframe

print(train_df.dshape)
print(train_df.peek())



weekly_mean = blz.by(train_df.Semana,return_mean=train_df.Demanda_uni_equil.mean(),sales_mean=train_df.Venta_uni_hoy.mean())