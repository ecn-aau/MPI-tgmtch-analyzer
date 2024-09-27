import os
import pandas as pd
import re

def lsb_read_data_in_dir(path, df_formatter=None, skip_first_row=False, prefix=None):
    df_list=[]
    for file in os.listdir(path):
        if ((prefix is None or file.startswith(prefix)) and (not os.path.isdir(path + "/" + file))):
            print(file)
            df_tmp = pd.read_table(path + "/" + file, comment='#', delim_whitespace=True)
            if (skip_first_row): df_tmp = df_tmp.iloc[1: , :] 
            if (df_formatter): df_formatter(df_tmp, file)
            df_list.append(df_tmp)  
    return pd.concat(df_list)

def msg_rate(df, small_win_size, big_win_size, key, calibrate=True):
    df_small_win = df[df["window_size"] == small_win_size]
    df_small_win_g = pd.DataFrame(df_small_win.groupby(key)[["time"]].median())
    df_big_win = df[df["window_size"] == big_win_size]
    df = pd.merge(df_big_win, df_small_win_g, on=key)
    if calibrate:
        df["time_cal"] = df["time_x"] - df["time_y"]
        df["msg_rate"] = (big_win_size-small_win_size)*1e6 / df["time_cal"]      
    else:
        df["time_cal"] = df["time_x"]
        df["msg_rate"] = (big_win_size)*1e6 / df["time_cal"]      
    df = df.reset_index()
    return df

    
#df = lsb_read_data_in_dir("data/pipeline_scalmatch_newqueue_aff_m10_t50/")
#df["receiver_type"] = "mpi"
#df = lsb_msg_rate(df, 32, 128, ["receiver_type", "msg_size", "match_threads", "rcv_idx"])