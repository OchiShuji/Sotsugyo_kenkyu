#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import os
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import cv2
import traceback


def gdata(dire, photo, control, indx, pix):
    img = Image.open("%s/photo/%s.JPG" % (dire, photo)).resize((int(10 * 25 / pix), int(6 * 25 / pix)))
    width, height = img.size
    imgdata = img.getdata()
    dataG_list = []
    dataG = pd.DataFrame(index=[], columns=["x", "y", "dist"])
    
    GR_mt = control.loc[indx, "GR_mt"]
    GR_lt = control.loc[indx, "GR_lt"]
    GG_mt = control.loc[indx, "GG_mt"]
    
    for y in tqdm(range(height)):
        ycache = y * width
        if int(height / 4) < y < int(3 * height / 4):
            continue
        for x in range(width):
            if int(width / 5) < x < int(4 * width / 5):
                continue
            pixdata = imgdata[ycache + x] # color_R = pixdata[0], color_G = pixdata[1], color_B = pixdata[2]
            if GR_mt <= pixdata[0] <= GR_lt and pixdata[1] >= GG_mt: #and color_G > pixdata[2] - 5:
                dataG_list.append((x, y, (x ** 2 + y ** 2) ** (1 / 2)))
    
    dataG = pd.DataFrame(dataG_list, columns=dataG.columns)
    kmG = KMeans(n_clusters=4, max_iter=100, random_state=1)
    clusterG = kmG.fit_predict(dataG)
    dataG['cluster_id'] = clusterG
    centersG = kmG.cluster_centers_
    # 3列目のdistを基準に行を入れ替え(ピクセル値は左上が原点)
    centersG = centersG[np.argsort(centersG[:, 2])]
    np.savetxt("%s/Gdata/greenplot_%s.csv" % (dire, photo), centersG, delimiter=",")

    plt.scatter(centersG[:, 0], centersG[:, 1], s=100, facecolor="none", edgecolors="blue", marker=".")
    plt.grid(True)
    plt.savefig(("%s/Gdata/Gplot_%s.png" % (dire, photo)))
    plt.close()

def rotationandcut(dire, photo, control, AoA, pix):
    img = cv2.imread("%s/photo/%s.JPG" % (dire, photo))
    img = cv2.resize(img, (int(10 * 25 / pix), int(6 * 25 / pix)))

    # 画像の切り出し
    raw_cp_data = np.loadtxt("%s/Gdata/greenplot_%s.csv" % (dire, photo), delimiter=",")
    pts1 = np.float32([[raw_cp_data[0, 0], raw_cp_data[0, 1]], [raw_cp_data[1, 0], raw_cp_data[1, 1]],
                       [raw_cp_data[2, 0], raw_cp_data[2, 1]], [raw_cp_data[3, 0], raw_cp_data[3, 1]]])
    pts2 = np.float32([[0, 0], [0, int(6 * 25 / pix)], [int(10 * 25 / pix), 0], [int(10 * 25 / pix), int(6 * 25 / pix)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (int(10 * 25 / pix), int(6 * 25 / pix)))

    center_x = control.loc[indx, "center_x"]
    a1 = math.sqrt(int(75 / pix) ** 2 + int((center_x * 200 + 25) / pix) ** 2)
    a2 = math.sqrt(int(75 / pix) ** 2 + int(((1 - center_x) * 200 + 25) / pix) ** 2)
    AoA = AoA * math.pi / 180
    if AoA >= 0:
        yp = a2 * math.cos(math.atan(((1 - center_x) * 200 + 25) / 75) - AoA)
        ym = a1 * math.cos(math.atan((center_x * 200 + 25) / 75) - AoA)
        xp = a2 * math.cos(math.atan(75 / ((1 - center_x) * 200 + 25)) - AoA)
        xm = a1 * math.cos(math.atan(75 / (center_x * 200 + 25)) - AoA)
    else:
        yp = a1 * math.cos(math.atan((center_x * 200 + 25) / 75) + AoA)
        ym = a2 * math.cos(math.atan(((1 - center_x) * 200 + 25) / 75) + AoA)
        xp = a2 * math.cos(math.atan(75 / ((1 - center_x) * 200 + 25)) + AoA)
        xm = a1 * math.cos(math.atan(75 / (center_x * 200 + 25)) + AoA)

    new_row = int(xp + xm)
    new_col = int(yp + ym)
    movex = xm - int((center_x * 200 + 25) / pix)
    movey = yp - int(75 / pix)
    D = np.float32([[1, 0, movex], [0, 1, movey]])
    dst = cv2.warpAffine(img, D, (new_row, new_col))
    M = cv2.getRotationMatrix2D((xm, yp), AoA * 180 / math.pi, 1)
    dst = cv2.warpAffine(dst, M, (new_row, new_col))

    cut_dst = dst[int(yp - 50 / pix):int(yp + 50 / pix), int(xm - (center_x * 200 + 12.5) / pix):int(xm + ((1 - center_x) * 200 + 12.5) / pix)]
    cv2.imwrite("%s/rotatecut/rotate_%s.JPG" % (dire, photo), dst)
    cv2.imwrite("%s/rotatecut/rotatecut_%s.JPG" % (dire, photo), cut_dst)

def spoit_red(dire, photo, pix, control, indx):
    img = Image.open("%s/rotatecut/rotatecut_%s.JPG" % (dire, photo))
    width, height = img.size
    imgdata = img.getdata()
    data_list = []
    dataR_list = []
    dataR = pd.DataFrame(index=[], columns=["x", "y"])

    RR_mt = control.loc[indx, "RR_mt"]
    RG_lt = control.loc[indx, "RG_lt"]
    RB_lt = control.loc[indx, "RB_lt"]

    for y in tqdm(range(height)):
        ycache = y * width
        for x in range(width):
            pixdata = imgdata[ycache + x] # color_R = pixdata[0], color_G = pixdata[1], color_B = pixdata[2]
            if pixdata[0] > RR_mt and pixdata[1] < RG_lt and pixdata[2] < RB_lt:
                dataR_list.append((x, y))
                data_list.append((255, 0, 0))
            else:
                data_list.append((255, 255, 255))
    
    dataR = pd.DataFrame(dataR_list, columns=dataR.columns)
    img.putdata(data_list)

    ## dataRはy軸方向を反転し、一般の座標系(単位pix)で表示
    dataR["y"] = height - dataR["y"]

    dataR.to_csv("%s/Rdata/redspoit_%s.csv" % (dire, photo))
    img.save("%s/Rdata/Rspoit_%s.jpg" % (dire, photo))

def plot_red(dire, photo, pix, control, indx):
    dataR = pd.read_csv("%s/Rdata/redspoit_%s.csv" % (dire, photo), index_col=0, parse_dates=True)

    # 前縁・後縁・天井・底の座標をピクセル値最大で指定
    LE_pix = dataR.loc[np.argmin(dataR["x"].values)]
    TE_pix = dataR.loc[np.argmax(dataR["x"].values)]
    TOP_pix = dataR.loc[np.argmin(dataR["y"].values)]
    BTM_pix = dataR.loc[np.argmax(dataR["y"].values)]

    # x方向分割間隔(pix)=10mm
    Xcut_int = int(2 / pix)
    Xcut_list = [LE_pix[0]]
    while Xcut_list[-1] < TE_pix[0] - Xcut_int:
        Xcut_list.append(Xcut_list[-1] + Xcut_int)
    else:
        Xcut_list.append(TE_pix[0])

    # y方向分割間隔(pix)
    #Ycut_int = int(10 / pix)
    #Ycut_list = [TOP_pix[1]]
    #while Ycut_list[-1] < BTM_pix[1] - Ycut_int:
    #    Ycut_list.append(Ycut_list[-1] + Ycut_int)
    #else:
    #    Ycut_list.append(BTM_pix[1])

    centersR = pd.DataFrame(index=[], columns=["x", "y"])

    for i in range(len(Xcut_list)):
        databox = dataR[(Xcut_list[i] - 5 <= dataR["x"]) & (dataR["x"] <= Xcut_list[i] + 5)]
        if len(databox) == 0 or len(databox) == 1:
            pass
        else:
            kmR = KMeans(n_clusters=2, max_iter=100, random_state=1)
            clusterR = kmR.fit_predict(databox)
            centerR = kmR.cluster_centers_
            #二つのcenterが10pix以上離れていないかった場合は平均値を採用
            if math.sqrt((centerR[0, 0] - centerR[1, 0]) ** 2 + (centerR[0, 1] - centerR[1, 1]) ** 2) < 10:
                centerR[0, 0] = (centerR[0, 0] + centerR[1, 0]) / 2
                centerR[0, 1] = (centerR[0, 1] + centerR[1, 1]) / 2
                centerR = np.delete(centerR, 1, axis=0)
            else:
                pass
            plt.scatter(centerR[:, 0], centerR[:, 1], s=100, facecolor="none", edgecolors="red", marker=".")
            centersR = np.vstack([centersR, centerR])

    #for i in range(len(Ycut_list)):
    #    databox = dataR[(Ycut_list[i] - 5 <= dataR["y"]) & (dataR["y"] <= Ycut_list[i] + 5)]
    #    if len(databox) == 0 or len(databox) == 1:
    #        pass
    #    else:
    #        kmR = KMeans(n_clusters=2, max_iter=100, random_state=1)
    #        clusterR = kmR.fit_predict(databox)
    #        centerR = kmR.cluster_centers_
    #        ##二つのcenterが10pix以上離れていないかった場合は平均値を採用
    #        if math.sqrt((centerR[0, 0] - centerR[1, 0]) ** 2 + (centerR[0, 1] - centerR[1, 1]) ** 2) < 10:
    #            centerR[0, 0] = (centerR[0, 0] + centerR[1, 0]) / 2
    #            centerR[0, 1] = (centerR[0, 1] + centerR[1, 1]) / 2
    #            centerR = np.delete(centerR, 1, axis=0)
    #        else:
    #            pass
    #        plt.scatter(centerR[:, 0], centerR[:, 1], s=100, facecolor="none", edgecolors="red", marker=".")
    #        centersR = np.vstack([centersR, centerR])

    # XcutとYcutで重複があるとスプラインでエラーを生じるので重複削除
    centersR = np.round(centersR.astype(np.double))
    centersR_new = np.unique(centersR, axis=0)

    np.savetxt("%s/Rdata/redplot_%s.csv" % (dire, photo), centersR_new, delimiter=",")
    plt.axes().set_aspect('equal')
    plt.savefig("%s/Rdata/Rplot_%s.png" % (dire, photo))
    plt.close()

def sort(dire, photo, control, indx):
    data = np.loadtxt("%s/Rdata/redplot_%s.csv" % (dire, photo), delimiter=",")

    t_edge = data[data.argmax(0)[0], :]
    l_edge = data[data.argmin(0)[0], :]

    morphing_edge = control.loc[indx, "morphing_edge"]

    if morphing_edge == 0:
        edge_error_x = t_edge[0] - 212.5 / pix
        edge_error_y = t_edge[1] - 50 / pix
    elif morphing_edge == 1:
        edge_error_x = l_edge[0] - 12.5 / pix
        edge_error_y = l_edge[1] - 50 / pix

    ##LE(TE)がずれたpix分平行移動。画面端から12.5mmの位置に来るはず
    data[:, 0] = data[:, 0] - edge_error_x
    data[:, 1] = data[:, 1] - edge_error_y

    ##LEを原点に
    data[:, 0] = data[:, 0] - 12.5 / pix
    data[:, 1] = data[:, 1] - 50 / pix

    t_edge = data[data.argmax(0)[0], :]
    l_edge = data[data.argmin(0)[0], :]

    base_x = control.loc[indx, "base_x"] * 200 / pix
    base_y = control.loc[indx, "base_y"] * 200 / pix

    if t_edge[0] - base_x != 0:
        t_angle_rad = math.atan((t_edge[1] - base_y) / (t_edge[0] - base_x))
        t_angle = 180. / math.pi * t_angle_rad
    elif t_edge[1] - base_y >= 0:
        t_angle = 90
    else:
        t_angle = -90

    zero = np.zeros((data.shape[0], 1))
    data = np.hstack((data, zero))
    if t_angle >= 0:
        flag = 1
    else:
        flag = 0

    for i in range(data.shape[0]):
        x = data[i, 0]
        y = data[i, 1]
        if x - base_x != 0:
            a_angle_rad = math.atan((y - base_y) / (x - base_x))
            a_angle = 180. / math.pi * a_angle_rad
        elif y - base_y >= 0:
            a_angle = -90
        else:
            a_angle = 90
        if flag == 1 and data[i, 0] > base_x and data[i, 1] >= base_y and a_angle >= t_angle:  # 第1象限edge以降(edge正)
            data[i, 2] = a_angle - t_angle
        elif flag == 1 and data[i, 0] > base_x and data[i, 1] >= base_y and a_angle < t_angle:  # 第1象限edge未満(edge正)
            data[i, 2] = 360 - (t_angle - a_angle)
        elif flag == 0 and data[i, 0] > base_x and data[i, 1] >= base_y:  # 第1象限(edge負)
            data[i, 2] = a_angle - t_angle
        elif data[i, 0] <= base_x and data[i, 1] >= base_y:  # 第2象限(共通)
            data[i, 2] = a_angle + 180 - t_angle
        elif data[i, 0] <= base_x and data[i, 1] <= base_y:  # 第3象限(共通)
            data[i, 2] = a_angle + 180 - t_angle
        elif flag == 1 and data[i, 0] > base_x and data[i, 1] <= base_y:  # 第4象限(edge正)
            if a_angle != t_angle: # 360°は0°とする
                data[i, 2] = a_angle + 360 - t_angle
            else:
                data[i, 2] = 0.
        elif flag == 0 and data[i, 0] > base_x and data[i, 1] <= base_y and a_angle < t_angle:  # 第4象限edge未満(edge負)
            data[i, 2] = a_angle + 360 - t_angle
        elif flag == 0 and data[i, 0] > base_x and data[i, 1] <= base_y and a_angle >= t_angle:  # 第4象限edge以降(edge負)
            data[i, 2] = a_angle - t_angle
        else:
            pass

    data = data[data[:, 2].argsort(), :]
    data = np.delete(data, 2, 1)
    data = np.vstack((data, t_edge))

    # 近すぎるセンターは割愛
    #list = []
    #for i in range(len(data)-2):
    #    length = math.sqrt((data[i+1,0]-data[i+2,0])**2 + (data[i+1,1]-data[i+2,1])**2)
    #    if length < 1 / pix:
    #        list.append(i+1)
    #    else:
    #        pass

    #data = np.delete(data,list,axis=0)

    np.savetxt("%s/sortspline/sort_%s.csv" % (dire, photo), data, delimiter=",")

def spline(dire, photo):
    data = np.loadtxt("%s/sortspline/sort_%s.csv" % (dire, photo), delimiter=",")

    # スプライン作成
    data = np.hstack((data, np.zeros((data.shape[0], 2))))
    l = 0
    for i in range(data.shape[0] - 1):
        d = math.sqrt((data[i + 1, 0] - data[i, 0]) ** 2 + (data[i + 1, 1] - data[i, 1]) ** 2)
        data[i + 1, 2] = d
        l = l + d
        data[i + 1, 3] = l

    # xにおいて媒介変数表示
    t = data[:, 3].tolist()

    x = data[:, 0].tolist()
    f = interp1d(t, x, kind="cubic")

    y = data[:, 1].tolist()
    g = interp1d(t, y, kind="cubic")

    tnew2 = [0.]
    itr = 5
    for i in range(data.shape[0] - 1):
        dist = data[i + 1, 3]
        while dist - tnew2[-1] > itr:
            tnew2.append(tnew2[-1] + itr)
        else:
            tnew2.append(dist)
    
    sp_data = np.array([f(tnew2), g(tnew2)])
    sp_data_rev = sp_data.transpose()
    ##200mm = 1
    sp_data_rev[:, :] = sp_data_rev[:, :] / (200 / pix)
    np.savetxt("%s/sortspline/splinedata_%s.csv" % (dire, photo), sp_data_rev, delimiter=",")
    #splineデータを正規化
    #spline_rev[:, :] = 1 / sum * spline_rev[:, :]
    #np.savetxt("%s/sortspline/spnormdata_%s.csv" % (dire, photo),spline_org, delimiter=",")

def camber(dire, photo):
    spdata = np.loadtxt("%s/sortspline/splinedata_%s.csv" % (dire, photo), delimiter=",")
    zero = np.zeros((spdata.shape[0], 2))
    spdata = np.hstack((spdata, zero))
    dist = 0
    spdata[0, 3] = dist
    for i in range(len(spdata) - 1):
        length = math.sqrt((spdata[i, 0] - spdata[i + 1, 0]) ** 2 + (spdata[i, 1] - spdata[i + 1, 1]) ** 2)
        spdata[i + 1, 2] = length
        dist = dist + length
        spdata[i + 1, 3] = dist

    #後縁
    cmb_te = spdata[0,0:2]

    #後縁から前縁にかけての外板の距離
    te2le = spdata[-1, 3] / 2

    #後縁からの距離が半分となる点をLEと定義
    idx = np.abs(spdata[:,3]-te2le).argmin()
    cmb_le = [spdata[idx,0],spdata[idx,1]]

    upper = spdata[0:idx+1,:]
    lower = spdata[idx:,:]
    lower = lower[::-1,:]
    dist = 0
    lower[0, 3] = dist
    lower[0, 2] = 0
    for i in range(len(lower) - 1):
        length = math.sqrt((lower[i, 0] - lower[i + 1, 0]) ** 2 + (lower[i, 1] - lower[i + 1, 1]) ** 2)
        lower[i + 1, 2] = length
        dist = dist + length
        lower[i + 1, 3] = dist


    camberdata = np.hstack([cmb_te,cmb_te,cmb_te])
    p = 0.05
    while p < te2le:
        up_idx = np.abs(upper[:,3]-p).argmin()
        up_p = upper[up_idx,0:2]
        low_idx = np.abs(lower[:, 3] - p).argmin()
        low_p = lower[low_idx, 0:2]
        camber_p = (up_p+low_p)/2
        p_data = np.hstack([up_p, low_p, camber_p])
        camberdata = np.vstack([camberdata,p_data])
        p = p+0.05
    else:
        l_data = np.hstack([cmb_le,cmb_le,cmb_le])
        camberdata = np.vstack([camberdata, l_data])

    #スプラインに悪影響を及ぼすので、LEに近すぎる点は削除
    if camberdata[-2,4] < 0.05:
        camberdata = np.delete(camberdata, -2, 0)
    else:
        pass

    #スプライン作成
    x = camberdata[::-1,4].tolist()
    y = camberdata[::-1,5].tolist()
    f = interp1d(x,y,kind="cubic")

    tnew = [x[0]]
    itr = 0.001
    for i in range(len(x) - 1):
        marker = x[i + 1]
        while marker - tnew[-1] > itr and tnew[-1] < x[-1] - itr:
            tnew.append(tnew[-1] + itr)
        else:
            tnew.append(marker)

    sp_data = np.array([tnew,f(tnew)]).transpose()

    # 前縁値を原点に移動
    le_org = camberdata[0, :]
    camberdata[:, 0] = camberdata[:, 0] - le_org[0]
    camberdata[:, 1] = camberdata[:, 1] - le_org[1]
    sp_data[:, 0] = sp_data[:, 0] - le_org[0]
    sp_data[:, 1] = sp_data[:, 1] - le_org[1]

    np.savetxt("%s/camber/camber_%s.csv" % (dire, photo), camberdata, delimiter=",")
    np.savetxt("%s/camber/cambersp_%s.csv" % (dire, photo), sp_data, delimiter=",")

    # chord lengthを1に正規化
    #sum = 0
    #for i in range(len(sp_data) - 1):
    #    length = math.sqrt((sp_data[i + 1, 0] - sp_data[i, 0]) ** 2 + (sp_data[i + 1, 1] - sp_data[i, 1]) ** 2)
    #    sum = sum + length

    #camberdata[:, :] = 1 / sum * camberdata[:, :]
    #sp_data[:, :] = 1 / sum * sp_data[:, :]

    #np.savetxt("%s/camber_norm/cambernorm_%s.csv" % (dire, photo), camberdata, delimiter=",")
    #np.savetxt("%s/camber_norm/camberspnorm_%s.csv" % (dire, photo), sp_data, delimiter=",")

def plot(dire, photo, control, AoA):
    spdata = np.loadtxt("%s/sortspline/splinedata_%s.csv" % (dire, photo), delimiter=",")
    camberdata = np.loadtxt("%s/camber/cambersp_%s.csv" % (dire, photo), delimiter=",")
    t_edge = spdata[spdata.argmax(0)[0], :]
    l_edge = spdata[spdata.argmin(0)[0], :]
    x = np.linspace(l_edge[0], t_edge[0], 100)
    y = np.linspace(l_edge[1], t_edge[1], 100)
    z = np.linspace(0, 1.0, 100)
    w = np.linspace(0, 0, 100)

    textsize = control.loc[indx, "textsize"]
    plt.figure(figsize=(10, 5))
    plt.plot(spdata[:, 0], spdata[:, 1])
    plt.plot(camberdata[:, 0], camberdata[:, 1], color='m')
    plt.plot(x, y, color='green')
    plt.plot(z, w, color='green')
    plt.axes().set_aspect('equal')
    plt.tick_params(labelsize = textsize)
    plt.xlabel("x/c", fontsize = textsize)
    plt.ylabel("y/c", fontsize = textsize)
    plt.grid(True)
    plt.ylim(-0.2, 0.2)
    plt.xlim(0, 1.0)
    plt.title("AoA = %s" % (AoA), fontsize = textsize)
    plt.savefig("%s/plot/plot_%s.png" % (dire, photo))
    plt.close()   

dire = os.getcwd()
control = pd.read_csv("%s/control.csv" % dire, delimiter=",")

if not os.path.exists("Gdata"):
    os.mkdir("Gdata")
if not os.path.exists("rotatecut"):
    os.mkdir("rotatecut")
if not os.path.exists("Rdata"):
    os.mkdir("Rdata")
if not os.path.exists("sortspline"):
    os.mkdir("sortspline")
if not os.path.exists("camber"):
    os.mkdir("camber")
if not os.path.exists("plot"):
    os.mkdir("plot")

for indx in range(len(control)):
    if control.loc[indx, "flag"] == 1:
        try:
            photo = control.loc[indx, "name"]
            pix = control.loc[indx, "pix"]
            AoA = control.loc[indx, "AoA"]
            print("%s is started" % photo)
            gdata(dire, photo, control, indx, pix)
            rotationandcut(dire, photo, control, AoA, pix)
            spoit_red(dire, photo, pix, control, indx)
            plot_red(dire, photo, pix, control, indx)
            sort(dire, photo, control, indx)
            spline(dire, photo)
            camber(dire,photo)
            plot(dire, photo, control, AoA)
            print("%s is finished" % photo)
        except:
            traceback.print_exc()
    else:
        pass

input("###########please enter any key#############")