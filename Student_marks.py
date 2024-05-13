import pandas as pd
import numpy as py
df1 = pd.read_csv('mark.csv')
df2 = pd.read_csv('student.csv')
# The two data will be merged together
combined_df = pd.merge(df1,df2)
#checking for any null or blank rows within the dataset, this is to sum the number of blank or null rows
combined_df.isnull().sum()
# Convert 'yes' to 1, other values to 0 in employed column
combined_df['Employed'] = combined_df['Employed'].apply(lambda x: 1 if x == 'yes' else 0)
# Define a mapping dictionary to classified grades 
grade_mapping = {'1st Class': 3, '2nd Class': 2, '3rd Class': 1}
# Apply the mapping to the Grade column, then create another column for numeric grade
combined_df['Numeric Grade'] = combined_df['Grade'].map(grade_mapping)