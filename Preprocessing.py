#%%
# Preprocessing Signals

import pandas as pd
import heartpy as hp
import neurokit2 as nk
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns



ppg_db_1 = pd.read_excel('full_raw_ppg_data.xlsx', sheet_name = 'ppg_db_1')

figure, axis = plt.subplots(3, 1, sharey=True)

axis[0, 1].plot[ppg_db_1['Red'].values]
axis[0, 1].set_title('Red PPG Waves')

axis[0, 2].plot[ppg_db_1['Green'].values]
axis[0, 2].set_title('Green PPG Waves')

axis[0, 3].plot[ppg_db_1['Blue'].values]
axis[0, 3].set_title('Blue PPG Waves')

plt.show()


# %%
