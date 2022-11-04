from glob import glob
import pandas as pd

output_files = glob('output_files/ac_*.csv')

file = pd.read_csv(output_files[0])

for file_name in output_files[1:]:
    file_tmp = pd.read_csv(file_name)
    file = pd.concat([file, file_tmp])

file = file[file['number_clusters'] != 0]
file.to_excel('output_files/summary_stats.xlsx', index=False)