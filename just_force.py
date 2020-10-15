#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

dir = os.getcwd()
data_dir = "%s/data" %dir

folder_list = os.listdir(data_dir)
print(folder_list)
folder_list = sorted(folder_list)
print(folder_list)
list1 = []
list2 = []
case_num = int(len(folder_list) / 2)

col_names = ["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]

data1_force = pd.DataFrame(index=[], columns=["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Fx_sd", "Fy_sd", "Fz_sd", "Mx_sd", "My_sd", "Mz_sd"])

def name_decide(filename):
    file = filename.split(".")[0]  #delete csv
    filelist = file.split("_")
    del filelist[:2]
    global new_name
    new_name = "_".join(filelist)
    return new_name

for i in range(case_num):
    data1_name = folder_list[2 * i]
    data1_org = pd.read_csv("%s/%s" % (data_dir, data1_name), delimiter=",", names=col_names, encoding="SHIFT-JIS")
    length = len(data1_org) - 4
    data1 = data1_org[4:len(data1_org)]
    data1 = data1.astype("float")
    data1_dsc = data1.describe()
    #name_decide(data1_name)
    case_list = [data1_dsc.loc["mean", "Fx"], data1_dsc.loc["mean", "Fy"], data1_dsc.loc["mean", "Fz"],
                 data1_dsc.loc["mean", "Mx"], data1_dsc.loc["mean", "My"], data1_dsc.loc["mean", "Mz"],
                 data1_dsc.loc["std", "Fx"], data1_dsc.loc["std", "Fy"], data1_dsc.loc["std", "Fz"],
                 data1_dsc.loc["std", "Mx"], data1_dsc.loc["std", "My"], data1_dsc.loc["std", "Mz"]]
    #data1_force.loc[new_name] = case_list
    data1_force.loc[data1_name] = case_list

# data2_force = pd.DataFrame(index=[],
#                            columns=["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Fx_sd", "Fy_sd", "Fz_sd", "Mx_sd", "My_sd",
#                                     "Mz_sd"])

# for i in range(case_num):
#     data2_name = folder_list[4 * i + 2]
#     data2_org = pd.read_csv("%s/%s" % (data_dir, data2_name), delimiter=",", names=col_names, encoding="SHIFT-JIS")
#     length = len(data2_org) - 4
#     data2 = data2_org[4:len(data1_org)]
#     data2 = data2.astype("float")
#     data2_dsc = data2.describe()
#     file = data2_name.split(".")[0]
#     casename = file.split("_")[2]
#     case_list = [data2_dsc.loc["mean", "Fx"], data2_dsc.loc["mean", "Fy"], data2_dsc.loc["mean", "Fz"],
#                  data2_dsc.loc["mean", "Mx"], data2_dsc.loc["mean", "My"], data2_dsc.loc["mean", "Mz"],
#                  data2_dsc.loc["std", "Fx"], data2_dsc.loc["std", "Fy"], data2_dsc.loc["std", "Fz"],
#                  data2_dsc.loc["std", "Mx"], data2_dsc.loc["std", "My"], data2_dsc.loc["std", "Mz"]]
#     data2_force.loc[casename] = case_list

data1_force.to_csv("av_Forces.csv")
# data2_force.to_csv("withwindVs.csv")

# data3_force = pd.DataFrame(index=[],
#                            columns=["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Fx_sd", "Fy_sd", "Fz_sd", "Mx_sd", "My_sd",
#                                     "Mz_sd"])
# for i in range(case_num):
#     orgname1 = data1_force.index[i]
#     orgname2 = data2_force.index[i]
#     case = orgname1.split("-")[0]
#     data3_force.loc[case] = data2_force[orgname2] - data1_force[orgname1]

# data3_force = data2_force - data1_force.values
#
# data3_force.to_csv("diffVs.csv")

# col_names = ["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
# data_org = pd.read_csv("%s/20180829_152252_test.csv" %data_dir, delimiter= ",", names=col_names, encoding="SHIFT-JIS")
# length = len(data_org) - 4
# data = data_org[4:len(data_org)]
#
#
#
# Fx = 0.
# Fy = 0.
# Fz = 0.
# Mx = 0.
# My = 0.
# Mz = 0.
#
#
# data = data.astype("float")
# print(data.describe())
