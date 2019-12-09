import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

df = pd.read_csv("diabetic_data.csv")

def printStats():
    print 'Total Instances and Attributes:'
    print df.shape
    print df.head(5)

def displayDups(data):
    duplicate_rows_df = data[data.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_df.shape)

# displays the missing values in each column
def displayMissingValues(data):
    print("missing values:", data.isnull().sum(axis=0))

def displayMissingValuesRows(data):
    print("missing values:", data.isnull().sum(axis=1))

# replaces all missing values containing a '?' to be nan
def setupMissingData(data):
    data.replace('?', np.nan, inplace=True)

def printColNames(data):
    for col in data.columns:
        print(col)

def findSingleDataCols(data):
    nunique = data.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    print cols_to_drop

def isUnique(data):
    print 'Total Unique Values:'
    print data.apply(pd.Series.nunique)

# function to count the 'No' in a column. This could be useful
# for tuning/trimming the model
# THIS IS BROKE
def howManyNoInRow(data):
    for col in data.columns:
        print(col, len(df[df[col] == 'No']))

# prints the numbers of re-admitted vs not re-admitted
def howManyReAdmitted():
    #df.set_index(df["readmitted"]).count(level="NO")
    print 'Re-Admitted:'
    print df['readmitted'].value_counts()

# prints how many NaN values in the whole data
def howManyNaN(data):
    print 'Total Missing Values:'
    print data.isnull().sum().sum()

#function to calculate how many 0's are in each colum
def howManyZeroInRow(data):
    for col in data.columns:
        print(col, len(df[df[col] == 0]))


#howManyReAdmitted()
#printStats()
print df.shape

# change the missing data to nan
setupMissingData(df)
# howManyNaN(df)
displayMissingValues(df)
# isUnique(df)

# display missing data and remove it, display it again to show change. Also drop encounter_no
# as it is completely unique and not relovent.
#displayMissingValues(df);
df.drop(columns=['weight', 'medical_specialty', 'payer_code', 'encounter_id'], axis=1, inplace=True)

# Check for high zero value columns as this could be a sign of noise
#howManyZeroInRow(df)
#df.drop(columns=['number_outpatient', 'number_emergency', 'number_inpatient'], axis=1, inplace=True)

# remove diagnosis 2 and 3 as they are not as critical as the first diagnosis
df.drop(columns=['diag_2', 'diag_3'], axis=1, inplace=True)
#printStats()

# drop rows with empty values
df = df.dropna()
#print df.shape

# check for columns with all the one value, then drop them
#findSingleDataCols(df)
df.drop(columns=['examide', 'citoglipton', 'metformin-rosiglitazone'], axis=1, inplace=True)
#print df.shape # show that the cols have droped by 3

# Remove the duplicate patient_no entries as we are only interested in the first admission
df = df.drop_duplicates(subset='patient_nbr', keep='first')
# Now remove the column as it has served it's purpose
df.drop(columns=['patient_nbr'], axis=1, inplace=True)
print df.shape
# Remove rows where the discharge_disposition_id is equal to 11, 13 and 14 because
# this means that they are going to die, also 19, 20 and 21
df = df.drop(df[df.discharge_disposition_id == 11].index)
df = df.drop(df[df.discharge_disposition_id == 13].index)
df = df.drop(df[df.discharge_disposition_id == 14].index)
df = df.drop(df[df.discharge_disposition_id == 19].index)
df = df.drop(df[df.discharge_disposition_id == 20].index)
df = df.drop(df[df.discharge_disposition_id == 21].index)
#print df.discharge_disposition_id

# replace the classifier column, "readmitted" values <30 and >30 with just "yes"
df['readmitted'] = df['readmitted'].replace(['<30', '>30'], 'YES')



# # change all catigorical values to binary values in new columns
# df = pd.get_dummies(df)
# #print df.dtypes

printStats()
