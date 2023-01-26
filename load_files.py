#%%
# Preprocessing and Feature Extraction
import pandas as pd
import os



#################################################################################################################################
#                                                                                                                               #
#           1) Load raw csv data into pd.DataFrame for compiling                                                                #
#                                                                                                                               #
#           2) Will form individual dataframes for each set of data for easier analysis later rather than opening each of       #
#           them individually.                                                                                                  #            
#                                                                                                                               #            
#           3) The individual dataframe are then saved on individual sheets of the same excel file: "full_raw_ppg_data.xlsx".   #
#           This will ensure all data is accessible from the same place                                                         #
#                                                                                                                               #
#################################################################################################################################
def get_ppg_files(path, writer):
    csvs = [x for x in os.listdir(path) if x.endswith('.csv')]
    csvs_list = list(csvs)
    print(csvs_list)
    for i in range(len(csvs_list)):
        dataframe, sheet_name = f'ppg_db_' + str(i), f'ppg_db_' + str(i)
        file = path + csvs_list[i]
        dataframe = pd.read_csv(file, index_col = 0).fillna(0, axis= 1)
        print(dataframe.head(2))
        dataframe.to_excel(writer, sheet_name=sheet_name)

             
if __name__ == "__main__":
    writer = pd.ExcelWriter('full_raw_ppg_data.xlsx', engine = "xlsxwriter")
    path = '/home/jazzy/Documents/PPS-Project/Raw_PPG_Files/'
    get_ppg_files(path=path, writer=writer)
    writer.close()
    
# %%
