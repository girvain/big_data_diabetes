from data_analysis_module import cleanData

import pandas as pd
import numpy as np

df = pd.read_csv("diabetic_data.csv")
# rum the clean data function to get a new filtered copy of the df
df_clean = cleanData(df)
#print df_clean.shape



