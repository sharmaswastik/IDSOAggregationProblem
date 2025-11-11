import os
import pandas as pd
import numpy as np
from glob import glob


# def excel_sheet_load(PLoad_dict, QLoad_dict):

#     cwd = os.getcwd()
#     path = r"OPF_data\load"

#     path = os.path.join(cwd, path)    
#     csv_files = glob(os.path.join(path, "*.csv"))

#     for f in csv_files:
#         df = pd.read_csv(f)
#         df = df.dropna(axis=1)
#         df = df.dropna(axis=0)
#         df.drop(df.index , inplace=True)
#         p_df = os.path.basename(f)
#         p = p_df.split('-')[-1].split('.')[0]
#         p = p.capitalize()
#         df.loc[0, 'Phase'] = p
#         if p_df[0] == 'P':
#             for key, value in Pload_dict.items():
#                 if key == p:
#                     for k, v in value.items():
#                         df['Phase'] = p
#                         df[k] = v
#                         df.to_csv(f, index = False)

#         else:
#             for key, value in Qload_dict.items():
#                 if key == p:
#                     for k, v in value.items():
#                         df['Phase'] = p
#                         df[k] = 0.99*a
#                         df.to_csv(f, index = False)
def excel_sheet_load(PLoad_dict, QLoad_dict):
    cwd = os.getcwd()
    path = r"OPF_data\load"
    path = os.path.join(cwd, path)    
    csv_files = glob(os.path.join(path, "*.csv"))

    for f in csv_files:
        df = pd.read_csv(f)
        df = df.dropna(axis=1)
        df = df.dropna(axis=0)
        df.drop(df.index , inplace=True)
        p_df = os.path.basename(f)
        p = p_df.split('-')[-1].split('.')[0]
        p = p.capitalize()

        # Ensure the DataFrame has at least one row
        if df.empty:
            df = pd.DataFrame({'Phase': [p]})

        df.loc[0, 'Phase'] = p

        if p_df[0] == 'P':
            for key, value in PLoad_dict.items():
                if key == p:
                    for k, v in value.items():
                        if isinstance(v, list):
                            if len(v) != len(df.index):
                                df = df.reindex(range(len(v)))
                        df['Phase'] = p
                        df[k] = v
                    df.to_csv(f, index=False)
        else:
            for key, value in QLoad_dict.items():
                if key == p:
                    for k, v in value.items():
                        if isinstance(v, list):
                            if len(v) != len(df.index):
                                df = df.reindex(range(len(v)))
                        df['Phase'] = p
                        if isinstance(v, list):
                            df[k] = [0.99 * val for val in v]
                        else:
                            df[k] = 0.99 * v
                    df.to_csv(f, index=False)
    
# Pload_dict = {'Bus0':50, 'Bus1':780, 'Bus2':850}
# Qload_dict = {'Bus0':0, 'Bus1':-13.8, 'Bus2':30.7}
# Pload_dict = {"A":{'Bus3':800, 'Bus48':960, 'Bus97':480, 'Bus117':800}, "B":{'Bus3':800, 'Bus48':960, 'Bus97':480, 'Bus117':800}, "C":{'Bus3':800, 'Bus48':960, 'Bus97':480, 'Bus117':800}}
Pload_dict = {"A":{'Bus0':0, 'Bus1':0, 'Bus2':0}, "B":{'Bus0':0, 'Bus1':0, 'Bus2':0}, "C":{'Bus0':0, 'Bus1':0, 'Bus2':0}}
Qload_dict = {"A":{'Bus0':0, 'Bus1':0, 'Bus2':0}, "B":{'Bus0':0, 'Bus1':0, 'Bus2':0}, "C":{'Bus0':0, 'Bus1':0, 'Bus2':0}}

# Pload_dict = {"A":{'Bus1':1100/3, 'Bus18':550/3, 'Bus23':550/3, 'Bus29':550/3, 'Bus66':550/3, 'Bus76':550/3, 'Bus105':550/3, 'Bus151':550/3}, "B":{'Bus1':1100/3, 'Bus18':550/3, 'Bus23':550/3, 'Bus29':550/3, 'Bus66':550/3, 'Bus76':550/3, 'Bus105':550/3, 'Bus151':550/3}, "C":{'Bus1':1100/3, 'Bus18':550/3, 'Bus23':550/3, 'Bus29':550/3, 'Bus66':550/3, 'Bus76':550/3, 'Bus105':550/3, 'Bus151':550/3}}
# Qload_dict = {"A":{'Bus1':1100/3, 'Bus18':550/3, 'Bus23':550/3, 'Bus29':550/3, 'Bus66':550/3, 'Bus76':550/3, 'Bus105':550/3, 'Bus151':550/3}, "B":{'Bus1':1100/3, 'Bus18':550/3, 'Bus23':550/3, 'Bus29':550/3, 'Bus66':550/3, 'Bus76':550/3, 'Bus105':550/3, 'Bus151':550/3}, "C":{'Bus1':1100/3, 'Bus18':550/3, 'Bus23':550/3, 'Bus29':550/3, 'Bus66':550/3, 'Bus76':550/3, 'Bus105':550/3, 'Bus151':550/3}}

# Pload_dict = {"A":{'Bus51':100, 'Bus105':40, 'Bus99':20, 'Bus18':150, 'Bus52':110, 'Bus1':160}, "B":{'Bus51':100, 'Bus105':40, 'Bus99':20, 'Bus18':150, 'Bus52':110, 'Bus1':160}, "C":{'Bus51':100, 'Bus105':40, 'Bus99':20, 'Bus18':150, 'Bus52':110, 'Bus1':160}}
# Qload_dict = {"A":{'Bus51':100, 'Bus105':40, 'Bus99':20, 'Bus18':150, 'Bus52':110, 'Bus1':160}, "B":{'Bus51':100, 'Bus105':40, 'Bus99':20, 'Bus18':150, 'Bus52':110, 'Bus1':160}, "C":{'Bus51':100, 'Bus105':40, 'Bus99':20, 'Bus18':150, 'Bus52':110, 'Bus1':160}}

# Pload_dict = {"A":{'Bus1':[1100/3, 550/3], 'Bus18':[550/3,550/3], 'Bus23':[550/3,550/3], 'Bus29':[550/3, 550/3], 'Bus66':[550/3, 550/3], 'Bus76':[550/3, 550/3], 'Bus105':[550/3, 550/3], 'Bus151':[550/3, 550/3]}, "B":{'Bus1':[1100/3, 550/3], 'Bus18':[550/3,550/3], 'Bus23':[550/3,550/3], 'Bus29':[550/3, 550/3], 'Bus66':[550/3, 550/3], 'Bus76':[550/3, 550/3], 'Bus105':[550/3, 550/3], 'Bus151':[550/3, 550/3]}, "C":{'Bus1':[1100/3, 550/3], 'Bus18':[550/3,550/3], 'Bus23':[550/3,550/3], 'Bus29':[550/3, 550/3], 'Bus66':[550/3, 550/3], 'Bus76':[550/3, 550/3], 'Bus105':[550/3, 550/3], 'Bus151':[550/3, 550/3]}}
# Qload_dict = {"A":{'Bus1':[1100/3, 550/3], 'Bus18':[550/3,550/3], 'Bus23':[550/3,550/3], 'Bus29':[550/3, 550/3], 'Bus66':[550/3, 550/3], 'Bus76':[550/3, 550/3], 'Bus105':[550/3, 550/3], 'Bus151':[550/3, 550/3]}, "B":{'Bus1':[1100/3, 550/3], 'Bus18':[550/3,550/3], 'Bus23':[550/3,550/3], 'Bus29':[550/3, 550/3], 'Bus66':[550/3, 550/3], 'Bus76':[550/3, 550/3], 'Bus105':[550/3, 550/3], 'Bus151':[550/3, 550/3]}, "C":{'Bus1':[1100/3, 550/3], 'Bus18':[550/3,550/3], 'Bus23':[550/3,550/3], 'Bus29':[550/3, 550/3], 'Bus66':[550/3, 550/3], 'Bus76':[550/3, 550/3], 'Bus105':[550/3, 550/3], 'Bus151':[550/3, 550/3]}}

excel_sheet_load(Pload_dict, Qload_dict)

