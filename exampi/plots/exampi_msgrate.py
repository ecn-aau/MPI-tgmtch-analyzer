import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

sys.path.append(os.getcwd() + "/plots")
from dpatm_utils import *

data_path="../data/exampi/"


data_mpi_nc_1 = pd.read_table(data_path + "mpi/mpi_nc_1.txt", comment='#', delim_whitespace=True)
data_mpi_nc_128 = pd.read_table(data_path + "mpi/mpi_nc_128.txt", comment='#', delim_whitespace=True)
data_mpi_nc = pd.concat([data_mpi_nc_1, data_mpi_nc_128])
data_mpi_nc["workload_type"] = "nc"
data_mpi_nc["matchers"] = 0

data_mpi_wc_1 = pd.read_table(data_path + "mpi/mpi_wc_1.txt", comment='#', delim_whitespace=True)
data_mpi_wc_128 = pd.read_table(data_path + "mpi/mpi_wc_128.txt", comment='#', delim_whitespace=True)
data_mpi_wc = pd.concat([data_mpi_wc_1, data_mpi_wc_128])
data_mpi_wc["workload_type"] = "wc"
data_mpi_wc["matchers"] = 0

data_devx_1 = pd.read_table(data_path + "devx/devx_1.txt", comment='#', delim_whitespace=True)
data_devx_128 = pd.read_table(data_path + "devx/devx_128.txt", comment='#', delim_whitespace=True)
data_devx = pd.concat([data_devx_1, data_devx_128])
data_devx["workload_type"] = "nc"
data_devx["matchers"] = 0

dpa_df_list=[]
for num_threads in [1, 2, 4, 8, 16, 32]:
    data_dpa_nc_1 = pd.read_table(data_path + "dpa/dpa_nc_" + str(num_threads) + "_1.txt", comment='#', delim_whitespace=True)
    data_dpa_nc_128 = pd.read_table(data_path + "dpa/dpa_nc_" + str(num_threads) + "_128.txt", comment='#', delim_whitespace=True)
    data_dpa_nc = pd.concat([data_dpa_nc_1, data_dpa_nc_128])
    data_dpa_nc["workload_type"] = "nc"
    data_dpa_nc["matchers"] = num_threads
    dpa_df_list.append(data_dpa_nc)   

    data_dpa_wc_sp_1 = pd.read_table(data_path + "dpa/dpa_wc_sp_" + str(num_threads) + "_1.txt", comment='#', delim_whitespace=True)
    data_dpa_wc_sp_128 = pd.read_table(data_path + "dpa/dpa_wc_sp_" + str(num_threads) + "_128.txt", comment='#', delim_whitespace=True)
    data_dpa_wc_sp = pd.concat([data_dpa_wc_sp_1, data_dpa_wc_sp_128])
    data_dpa_wc_sp["workload_type"] = "wc_sp"
    data_dpa_wc_sp["matchers"] = num_threads
    dpa_df_list.append(data_dpa_wc_sp)   

    data_dpa_wc_fp_1 = pd.read_table(data_path + "dpa/dpa_wc_fp_" + str(num_threads) + "_1.txt", comment='#', delim_whitespace=True)
    data_dpa_wc_fp_128 = pd.read_table(data_path + "dpa/dpa_wc_fp_" + str(num_threads) + "_128.txt", comment='#', delim_whitespace=True)
    data_dpa_wc_fp = pd.concat([data_dpa_wc_fp_1, data_dpa_wc_fp_128])
    data_dpa_wc_fp["workload_type"] = "wc_fp"
    data_dpa_wc_fp["matchers"] = num_threads
    dpa_df_list.append(data_dpa_wc_fp)   

data_dpa = pd.concat(dpa_df_list)
data_dpa["receiver_type"] = "dpa"
print(data_dpa)

data_baselines = pd.concat([data_mpi_nc, data_mpi_wc, data_devx])
key=["workload_type", "msg_size", "runs", "recv_index", "matchers", "receiver_type"]
#df_rate_baselines = msg_rate(data_baselines, 1, 128, key, True)

df_rate_mpi_wc = msg_rate(data_mpi_wc, 1, 128, key, True)
df_rate_mpi_nc = msg_rate(data_mpi_nc, 1, 128, key, True)
df_rate_devx = msg_rate(data_devx, 1, 128, key, True)
df_rate_dpa = msg_rate(data_dpa, 1, 128, key, True)


df_rate_avg_mpi_wc = df_rate_mpi_wc["msg_rate"].median()
df_rate_avg_mpi_nc = df_rate_mpi_nc["msg_rate"].median()
df_rate_avg_devx = df_rate_devx["msg_rate"].median()

#scal_plot = sns.lineplot(data=df_rate_dpa, x="matchers", y="msg_rate", hue="workload_type", marker="o", estimator=np.median)
#scal_plot.set(xlabel='DPA threads', ylabel='Message rate (Mmps)')
#scal_plot.axhline(y=df_rate_avg_mpi_wc)
#scal_plot.axhline(y=df_rate_avg_mpi_nc)
#scal_plot.axhline(y=df_rate_avg_devx, linestyle="dashed")
#plt.xscale('log', base=2)
#plt.show()

df_rate_dpa_32 = df_rate_dpa[df_rate_dpa["matchers"]==32]
df_rate_mpi = pd.concat([df_rate_mpi_wc, df_rate_mpi_nc])
#df_bplot = pd.concat([df_rate_dpa_32, df_rate_devx, df_rate_mpi_wc, df_rate_mpi_nc])
bplot = sns.barplot(x='workload_type', y='msg_rate', data=df_rate_dpa_32)
plt.show()

bplot = sns.barplot(x='workload_type', y='msg_rate', data=df_rate_mpi)
plt.show()

bplot = sns.barplot(x='workload_type', y='msg_rate', data=df_rate_devx)
plt.show()

print(df_rate_dpa)

