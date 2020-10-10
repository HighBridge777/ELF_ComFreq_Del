# coding=utf-8
import math
import numpy as np
from matplotlib import pyplot
from scipy.optimize import curve_fit
import scipy.optimize as optimize
import os
import csv
import sys
import time
import scipy.fftpack
##########ファイル読み込みクラス#################################
class ReadFile:
    #コンストラクタ部分
    def __init__(self,filename):
        #インタラクティブでファイル名を入力した値が入る
        self.__filename = filename
        #第一カラムと第二カラムを分ける
        self.__in_data_x = []
        self.__in_data_y = []
        #csv読み込んでx,y軸をパース
    def readCSV(self):
        print('csvファイルを読込み処理開始')
        try:
            # utf-8 のCSVファイル
            with open(self.__filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                #for文で第一カラムと第二カラムを取得する
                for row in csv_reader:
                    self.__in_data_x.append(float(row[0]))
                    self.__in_data_y.append(float(row[1]))
        # 起こりそうな例外をキャッチ
        except FileNotFoundError as e:
            print(e)
        except csv.Error as e:
            print(e)
        print('CSV読込み処理完了:\n')
        return self.__in_data_x,self.__in_data_y
'''
FittingDataクラスは主にオーバーラップ処理とフィッティング処理を担当
'''
class FittingData:
##################################コンストラクタ部分#########################
    def __init__(self,fHum,fSampling,sectionLength,n,overLapPer):
        #商用周波数[Hz]
        self.fHum = fHum
        #オフセット
        self.offset = 0
        #周波数のブレの幅[%],50Hz(60Hz)に対して何%か
        self.fWidthPer = 1
        #周波数の下限、上限の計算
        self.fMin = fHum - fHum * self.fWidthPer / 100
        self.fMax = fHum + fHum * self.fWidthPer / 100
        #データサンプリング[Hz]
        self.fSampling = fSampling
        #フィッティング期間[cycle]
        self.sectionLength = sectionLength
        #データ入力制限
        self.n = n
        #フィッティング配列
        self.totalArray = np.array([])
        #オーバーラップの割合
        self.overLapPer = overLapPer
        #一区間のデータ点数の計算
        self.sectionDataNum = math.ceil((fSampling / fHum * sectionLength))
        #オーバーラップするデータ点数の計算
        self.overLapDataNum = math.ceil((self.sectionDataNum * overLapPer / 100))
        #フレーム分割数
        #totalSectionNum = WorksheetFunction.RoundDown(n / (sectionDataNum - overLapDataNum), 0) - 1 なのにこちらではciel
        self.totalSectionNum = int((n / (self.sectionDataNum - self.overLapDataNum))- 1)
    ##########フィッティング関数#################################
    '''
    a:振幅
    b:周波数
    x:時間
    c:定数（オフセットで使用）
    p:位相
    '''
    ############################################################
    #定数にしていた位相を変数に変更(np.pi=>p)
    @classmethod
    def fittingFunc(cls,x,a,b,c,p):
        #フィッティングにおいて当てはめに使う関数
        y = a * np.sin(2*np.pi*b * x + p) + c
        return y
    ############################################################
    #RMSEの関数を定義
    @classmethod
    def rmse_res(cls,param,x,y):
        residual = y - FittingData.fittingFunc(x,param[0],param[1],param[2],param[3])
        resid_=np.sqrt(((residual*residual).sum())/len(x))
        return resid_
    ##########振幅の初期値の計算##################################
    @classmethod
    def calcA0(cls,aMin, aMax, vP ):
        CalcA0 = 0
        if vP > aMax:#上限を超えた場合
            CalcA0 = aMax
            return CalcA0
        elif vP < aMin: #下限を超えた場合
            CalcA0 = aMin
            return CalcA0
        else:#それ以外
            CalcA0 = vP
            return CalcA0
    ##########オーバーラップの関数#################################
    def ov(self,data):
        #データ長[s]
        Ts = len(data) / self.fSampling
        #フレーム周期[s]
        Fc = self.sectionDataNum / self.fSampling
        #オーバーラップ位置
        x_ol = self.sectionDataNum * (1 - (self.overLapPer/100))
        #分割数
        N_ave = int((Ts - (Fc * (self.overLapPer/100))) / (Fc * (1-(self.overLapPer/100))))
        #オーバーラップされた波形データの配列
        array = []
        #分割数分ループしてデータを入れる
        for i in range(N_ave):
            ps = int(x_ol * i)
            array.append(data[ps:ps+self.sectionDataNum:1])
        return array, N_ave
    ##########フィッティングメソッド###############################
    def curveFitting(self,N_ave,data_ol,data_ol2,fHum):
        pre_fit_y = np.array([])
        for i in range(N_ave):
            aMin = 0.01#振幅の下限（ｎT)
            aMax = 0.1#振幅の上限（ｎT)
            totalABSv1 = sum(data_ol[i])#一区間のvの絶対値のトータル[nT]
            v1Peak = totalABSv1 / self.sectionDataNum * math.sqrt(2)#一区間のv1のPeak値[nT]
            # |--振幅の初期値の場合分け
            a0 = FittingData.calcA0(aMin, aMax, v1Peak)#振幅aの初期値
            #パラメータ取得用配列（引数に初期設定）
            para = [a0, fHum, data_ol[i][0], 0]
            #パラメータの制限
            bnds = ((None, None), (self.fMin, self.fMax), (None, None), (-2 * np.pi, 2 * np.pi))
            #パラメータ取得
            mi_rmse=optimize.minimize(FittingData.rmse_res, para, args=(data_ol2[i], data_ol[i]),bounds=bnds)
            fit_y = FittingData.fittingFunc(data_ol2[i],mi_rmse.x[0],mi_rmse.x[1],mi_rmse.x[2],mi_rmse.x[3])
            #オフセット設定
            self.offset = mi_rmse.x[2]
            fit_y -= self.offset
            #オーバーラップの平均化処理
            if i > 0:
                ave =(pre_fit_y[self.sectionDataNum - self.overLapDataNum:self.sectionDataNum] + fit_y[0:self.overLapDataNum])/2
                fit_y[0:self.overLapDataNum] = ave
            #オーバーラップ分差し引いて処理
            if i == 0:
                self.totalArray = np.append(self.totalArray,fit_y[0:self.sectionDataNum])
            else:
                self.totalArray = np.append(self.totalArray,fit_y[self.overLapDataNum:self.sectionDataNum])
            #オーバーラップ処理で一つ前のデータが必要なのでfit_yの手前のデータという意味でpre_fit_y
            pre_fit_y = fit_y
##########グラフ化クラス##################################
class Graph:
    #コンストラクタ部分
    def __init__(self,name,x_axis,y_axis,x_label,y_label):
        self.name = name
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_label = x_label
        self.y_label = y_label
    def makePlot(self):
        print(self.name + 'データをグラフ化します')
        pyplot.xlabel(self.x_label, fontsize=14)
        pyplot.ylabel(self.y_label, fontsize=14)
        pyplot.plot(self.x_axis, self.y_axis)
        pyplot.show()
        print(self.name + 'グラフ化完了')
#####################################メイン処理###############################
def main():
    ###############################フィッティング用パラメータ設定####################
    '''7/06仕様にある、コマンドライン引数で周波数とオーバーラップ率を設定する仕様に変更
    例：周波数50Hzでオーバーラップ率25%を設定の場合 => denoise.py 50 25
    コマンドライン引数を入れない場合のデフォルト値は50Hz,25%の設定
    '''
    #周波数[Hz]
    fHum = 0.0
    #オーバーラップ率[%]
    overLapPer =0.0
    #コマンドライン引数を配列argsに代入
    args = sys.argv
    #コマンドライン引数で周波数とオーバーラップ率を設定しば場合
    if len(args)==3:
        #商用周波数[Hz]
        fHum = float(args[1])
        #オーバーラップの割合[%]
        overLapPer = float(args[2])
    #コマンドライン引数で周波数とオーバーラップ率を設定しない場合(デフォルト値)
    else:
        fHum = 50
        overLapPer = 25
    #周波数のブレの幅[%],50Hz(60Hz)に対して何%か
    fWidthPer = 1
    #周波数の下限、上限の計算
    fMin = fHum - fHum * fWidthPer / 100
    fMax = fHum + fHum * fWidthPer / 100
    #データサンプリング[Hz]
    fSampling = 8192
    #フィッティング期間[cycle]
    sectionLength = 0.75
    #データ入力制限
    n = 10000
    #相対時刻[sec],データ全点
    t = []
    #信号強度[nT],データ全点
    v = []
###################################ファイル読み込み##############################
    while True:
            print('解析するファイル名を入力してください（拡張子のところまで入力してください)')
            print('例：test.csv')
            print('*アプリを終了したい場合は [q] を押してください\n')
            filePath = input('>>')
            if filePath == 'q':
                sys.exit('アプリを正常終了します')
            else:
                path,ext = os.path.splitext(filePath)#ファイル名と拡張子に分ける
                if ext == '.csv':
                    csvfile = ReadFile(filePath)
                    t,v = csvfile.readCSV()#戻り値が２つ（x軸、y軸）
                    break
                else:
                    print('\n\n\n現時点ではCSVファイルのみの対応です\n\n\n')
                    continue

################################ノイズ抽出プログラム設定##########################
    print('ノイズ抽出処理開始')
    #処理開始時刻のセット
    start = time.time()
    #フィッティング計算用に元データ（時間、波形）をｘ、ｙ軸に読み込む
    np_x = np.array(t)
    np_y = np.array(v)
    #フィッティング用にインスタンス生成
    fittingdata = FittingData(fHum,fSampling,sectionLength,n,overLapPer)
    #####################オーバーラップ処理##################
    #オーバーラップ処理した分割数をN_aveに入れる
    data_ol, N_ave = fittingdata.ov(np_y)
    #オーバーラップ処理した分割数をN_aveに入れる
    data_ol2, N_ave2 = fittingdata.ov(np_x)
    #フィッティング結果のSinの値[nT],データ全点
    s = np.array([])
    np_zt = np.array([])
    #フィッティング処理
    fittingdata.curveFitting(N_ave,data_ol,data_ol2,fHum)
    #オーバーラップする分だけ前から取り出す
    #配列の取り出しの修正(CVE様ご指摘)
    count = 0
    for i in data_ol:
        if count == 0:
            s = np.append(s,i[:fittingdata.sectionDataNum])
        else:
            s = np.append(s,i[fittingdata.overLapDataNum:fittingdata.sectionDataNum])
        count +=1
    count = 0
    #配列の取り出しの修正
    for i in data_ol2:
        if count == 0:
            np_zt = np.append(np_zt,i[:fittingdata.sectionDataNum])
        else:
            np_zt = np.append(np_zt,i[fittingdata.overLapDataNum:fittingdata.sectionDataNum])
        count += 1

    #生データからフィッティングしたデータを引いてノイズ除去
    vNR = s- fittingdata.totalArray#ノイズ除去

    #ノイズ抽出の終了（現在ー開始を差し引く：計算自体はここで終わり、あとはグラフの印字のみ
    print('ノイズ抽出処理完了 :', time.time() - start, '[sec]')
    ########################グラフ化##############################################
    #元データ可視化
    rawData = Graph('生データ',np_x,np_y,'time(sec)','signal')
    rawData.makePlot()
    #オーバーラップ処理後のデータ
    overLappingGraph = Graph('オーバーラッピング結果',np_zt,s,'time(sec)','signal')
    overLappingGraph.makePlot()
    #フィッティングデータ可視化
    fittingGraph = Graph('フィッティング結果',np_zt,fittingdata.totalArray,'time(sec)','signal')
    fittingGraph.makePlot()
    #ノイズ除去データ可視化
    deNoiseGraph = Graph('ノイズ除去結果',np_zt,vNR,'time(sec)','signal')
    deNoiseGraph.makePlot()

    ###########ノイズ除去結果をCSVに吐き出す######################################
    with open('ノイズ除去結果.csv','w',newline="") as f:
        index=0
        writer = csv.writer(f)
        writer.writerow(['経過時間','ノイズ除去信号(nT)','フィッティングデータ','元データ'])
        for i in vNR:
            writer.writerow([np_x[index],i,fittingdata.totalArray[index],s[index]])
            index+=1
            #10000行までの出力
            if index >= (fittingdata.sectionDataNum-fittingdata.overLapDataNum)*(fittingdata.totalSectionNum-1):
                break
if __name__ == "__main__":
    main()
