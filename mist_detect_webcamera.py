"""
Created on Fri Dec  4 16:32:40 2020

@author: IMANAKA_HIDEO
"""
from locale import strxfrm
import tkinter
import tkinter.filedialog
from PIL import Image, ImageTk
import cv2
import time
from time import sleep
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter.font import Font
import datetime
import threading
from shot_102_driver import shot_102




class Image_Proc(threading.Thread):
    def __init__(self):

        threading.Thread.__init__(self)
        self.create_img_list()
        self.initial_HSV()
        
    def initial_HSV(self):
        '霧検出閾値の初期値設定'
        
        #下限閾値 S…90
        self.lowerH = 30
        self.lowerS = 65
        self.lowerV = 10
        
        #上限閾値
        self.upperH = 90
        self.upperS = 255
        self.upperV = 255

    def create_img_list(self):
        #データ保存リストの初期化
        self.ymax_list = []
        self.realheight_list = []
        self.area_sum_list = []
        self.area_pxrate_list = []
        self.timelist1 = []
        self.height_valuelist = []

        #変数の初期化v
        self.maxNum = 0
        self.bottom_height = 102
        #top=207cm
        #old 103cm
        self.px_cmValue = 0.229
        #old 0.2145cm/px
        #offset=1.5cm
        #測りの高さ入れる変数
        self.offset = 0
    
    def detect_height(self,frame):
        '輪郭の検出'
        
        self.frame = frame

        frame_copy = self.frame.copy()
        
        #メディアンフィルタ
        frame_copy = cv2.medianBlur(frame_copy,5)
        #HSV変換
        hsv = cv2.cvtColor(frame_copy,cv2.COLOR_BGR2HSV)
        
        #HSVに変換したフレームから緑色の部分のみの抽出
        lower_color = np.array([self.lowerH,self.lowerS,self.lowerV])
        upper_color = np.array([self.upperH,self.upperS,self.upperV])     
        self.mask = cv2.inRange(hsv,lower_color,upper_color)
        self.output = cv2.bitwise_and(hsv,hsv,mask=self.mask)
        
        #緑部分を抽出したフレームをグレースケールに変換
        gray = cv2.cvtColor(self.output,cv2.COLOR_BGR2GRAY)  
        #クロージング処理で緑部分を大きくする
        kernel_Close = np.ones((7,7),np.uint8)
        gray_close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel_Close)

        #オープニング処理でノイズの除去
        kernel_Open = np.ones((3,3),np.uint8)
        gray_open = cv2.morphologyEx(gray_close,cv2.MORPH_OPEN,kernel_Open)
        
        #モルフォロジー処理した画像から輪郭の検出
        contours,hierarchy = cv2.findContours(gray_open,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        img = cv2.drawContours(frame,contours,-1,(255,0,0),3)  
        
        self.con=contours
        
        return img

    def calc_height(self):
        '高さの計算'
        
        con = self.con
        area=0
        x_list = []
        y_list = []
        area_list = []
        #現在のフレームから検出された輪郭をx,y座標ごとに配列に格納
        for i in range(len(con)):
            buf_np = con[i].squeeze(axis=1).ravel()
            area = area+cv2.contourArea(con[i])
            
            for i,elem in enumerate(buf_np):

                if i%2 == 0:
                    x_list.append(elem)

                else:                   
                    y_list.append(elem*(-1))

        self.area_sum_list.append(area)   

        #現在のフレームのy座標のリストから最大値を取り出す・輪郭ない場合は０にする         
        if not y_list:
            self.ymax = -self.height
            self.ymax_real = self.bottom_height
            
        else:
            self.ymax = max(y_list)+self.height
            self.ymax_real = self.convert_px()
        
        if self.maxNum < self.ymax_real:
            self.maxNum = self.ymax_real
            self.maximg = self.output
            self.maximg_array = np.array(self.maximg)
            self.img_max_hist = cv2.calcHist([self.maximg],[0],self.mask,[256],[0,256])
            self.img_max_hist2 = cv2.calcHist([self.maximg],[1],self.mask,[256],[0,256])
            self.img_max_hist3 = cv2.calcHist([self.maximg],[2],self.mask,[256],[0,256])
            
        self.ymax_list.append(self.ymax)        
        self.realheight_list.append(self.ymax_real)
        self.ave_realheight = sum(self.realheight_list)/len(self.realheight_list)

    def create_hist_graph(self):
        #噴霧高最大時のフレームのHSV平均値の算出
        h=self.maximg.T[0].ravel().mean()
        s=self.maximg.T[1].ravel().mean()
        v=self.maximg.T[2].ravel().mean()

        #ヒストグラムの作成
        fig_hist = plt.figure()
        
        ax = fig_hist.add_subplot(111)
        
        ax.set_xlabel('time(frame)',fontsize=10)        
        ax.set_ylabel('pixel_counts',fontsize=10)
        ax.plot(self.img_max_hist,c = "red",label = "H")
        ax.plot(self.img_max_hist3,c = "blue",label = "S")
        #plt.savefig('./result/'+'maxHeightHist'+self.time_now.strftime('%Y%m%d%H%M') + '.png')

    def convert_px(self):
        #real_height=(self.ymax*0.1791)+100
        #cm換算式① メジャーの上下限のメモリから縦の大きさを割り出してcm換算
        #real_height = (self.ymax*self.px_cmValue)+self.bottom_height+self.offset
        #real_height = (0.9953*real_height)-1.573

        #メジャーの10cm間隔の目盛りの座標からcm換算
        real_height = (0.2279*self.ymax)+98.456

        #cm換算値誤差の計算
        mes_error = (0.05*real_height)-8

        #誤差の修正
        real_height = real_height-mes_error
        
        #高すぎるところの輪郭はノイズであるため小さい数値にする
        if real_height >= 205:
            
            real_height = 150
        
        return real_height

    def create_graph(self):
        'グラフの作成'
        
        newWindow = tkinter.Toplevel()
        
        fig = plt.figure()
        
        height = fig.add_subplot(2,2,1)
        height.set_xlabel('time(frame)',fontsize=10)        
        height.set_ylabel('Height(cm)',fontsize=10)
        height.plot(self.realheight_list)
        area = fig.add_subplot(2,2,2)
        area.set_xlabel('time(frame)',fontsize=10)
        area.set_ylabel('area(px^2)',fontsize=10)
        area.plot(self.area_sum_list)
        canvas = FigureCanvasTkAgg(fig, newWindow)        
        canvas.draw()
        
        newWindow = None
        
        canvas.get_tk_widget().pack() 
        
        if os.path.exists("./result") == False:
            os.mkdir("./result")
            
        #plt.savefig('./result/'+'resultGraph'+self.time_now.strftime('%Y%m%d%H%M')+'.png')

    def create_img_data(self):
        '画像配列をCSVに保存'
        self.maximg_array = self.maximg_array.ravel()
        maximg_array_df = pd.Series(self.maximg_array)
        hist_h_df = pd.Series(self.img_max_hist.ravel())
        hist_s_df = pd.Series(self.img_max_hist2.ravel())
        hist_v_df = pd.Series(self.img_max_hist3.ravel())

        img_DF=pd.concat((maximg_array_df.rename(r'#maximg_px'),
                        hist_h_df.rename('hist_H'),
                        hist_h_df.rename('hist_S'),
                        hist_v_df.rename('hist_V'),),
                        axis=1,sort=False
        )
        if os.path.exists("./result")==False:
            os.mkdir("./result")
        
        #img_DF.to_csv('./result/'+self.time_now.strftime('%Y%m%d%H%M')+'_Imgdata.csv', encoding="utf-8",index=False)
        #cv2.imwrite('./result/'+'maxHeight'+self.time_now.strftime('%Y%m%d%H%M')+'.png',self.frame)        
        
    def create_logfile(self):
        'CSVファイル作成・高さデータの書き込み'
        
        ymax_df = pd.Series(self.ymax_list)
        RH_df = pd.Series(self.realheight_list)   
        area_df = pd.Series(self.area_sum_list)
        maxheight_df = pd.Series(self.maxNum)
        aveheight_df = pd.Series(self.ave_realheight)

        #高さのリストから高さの頻度計算
        list = [int(n) for n in self.realheight_list]
        over180 = sum(i >= 180 for i in list)
        over170 = sum(i >= 170 and i < 180 for i in list)
        over160 = sum(i >= 160 and i < 170 for i in list)
        over150 = sum(i >= 150 and i < 160 for i in list)
        over140 = sum(i >= 140 and i < 150 for i in list)
        over130 = sum(i >= 130 and i < 140 for i in list)
        over120 = sum(i >= 120 and i < 130 for i in list)
        over110 = sum(i >= 110 and i < 120 for i in list)
        
        frequency_height = []
        
        frequency_height.append(over180)
        frequency_height.append(over170)
        frequency_height.append(over160)
        frequency_height.append(over150)
        frequency_height.append(over140)
        frequency_height.append(over130)
        frequency_height.append(over120)
        frequency_height.append(over110)

        count_list = ['180以上','180未満170以上','170未満160以上','160未満150以上','150未満140以上',
        '140未満130以上','130未満120以上','120未満110以上']

        count_df = pd.Series(count_list)
        frequency_df = pd.Series(frequency_height)
        #各フレームを読み込んで描画し終わるまでにかかる時間の計算
        proctime_df = pd.Series(self.timelist1)
        
        DF = pd.concat((ymax_df.rename(r'#高さ(px)'),
                        RH_df.rename('高さ(cm)'),
                        area_df.rename('面積(px^2)'),
                        maxheight_df.rename('最大高さ(cm)'),
                        aveheight_df.rename('平均高さ(cm)'),
                        proctime_df.rename('処理時間(t/f)'),
                        count_df.rename('ヒストグラム(高さ計測フレーム数)'),
                        frequency_df.rename(''),),
                        axis=1,sort = False
                    )
        
        if os.path.exists("./result") == False:
            os.mkdir("./result")
        
        DF.to_csv('./result/'+self.time_now.strftime('%Y%m%d%H%M')+'CS1_HxxTxxCxxH.csv', encoding="utf-8_sig",index=False)
        
    def proctime(self,time):
        print(time)
        self.timelist1.append(time)
        #print(self.timelist1)

    def write_log(self):
        height_px = str(self.ymax)
        height_cm = str(self.ymax_real)
        px = "h(px)"
        cm = "h(cm)"
        time = "t(f)"            
        frame = str(len(self.ymax_list))
        send_log = time+"|"+frame+" "+px+"|"+height_px+" "+cm+"|"+height_cm
            
        return send_log

    def rec_video(self,fps,width,height):
        'ビデオファイル初期設定'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.time_now = datetime.datetime.now()
        
        if os.path.exists("./result") == False:
            os.mkdir("./result")
        
        self.out=cv2.VideoWriter('./result/'+'mistdetect'+self.time_now.strftime('%Y%m%d%H%M%S')+'.avi',
                                fourcc,fps,(width,height),True)

    def video_state(self,fps,width,height):
        self.fps = fps
        self.width = width
        self.height = height
        print('wid',self.width)
        print('height',self.height)
    #輪郭検出終了時の動作設定
    def rec_fin(self):
        self.out.release()
        self.create_logfile()
        self.create_hist_graph()
        self.create_graph()
        self.create_img_data()

class Model(threading.Thread):

    def __init__(self,imgproc):
        threading.Thread.__init__(self)
        self.imgproc = imgproc
        # 動画オブジェクト参照用
        self.initial_set()
        
    def initial_set(self):
        self.video = None
        # 画像処理の設定
        self.gray = False
        
        self.flip= False

        self.contour = False
        # 読み込んだフレーム
        self.frames = None
        # PIL画像オブジェクト参照用
        self.image = None
        # Tkinter画像オブジェクト参照用
        self.image_tk = None

    def create_video(self, path):
        if path != 'web':
            #pathの動画から動画オブジェクト生成
            self.video = cv2.VideoCapture(path)
        else:
        ################################################################################################################
        ##webカメラ接続・解像度などの設定###############################################################################
        ################################################################################################################
            self.video = cv2.VideoCapture(1) #webカメラ使用時
            #self.video = cv2.VideoCapture(0)  #pc内蔵カメラ使用時
            self.video.set(cv2.CAP_PROP_BUFFERSIZE,1)
            self.video.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
            #self.video.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
            #self.video.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
            self.imgproc.video_state(self.fps,self.width,self.height)
            #print(self.height)
        self.video_open = True
            
        if not self.video.isOpened():
            self.video.release()
            self.video_open = False
            
    def advance_frame(self):
        'フレームを読み込んで１フレーム進める'
        if not self.video:
            return
        # フレームの読み込み
        ret, self.frame = self.video.read()

        return ret
    
    def reverse_video(self):
        '動画を先頭に戻す'

        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def release_video(self):
        self.video.release()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cv2.destroyAllWindows()
        print(self.video.isOpened())
        
        if not self.video:
            return
        
    def create_image(self, size):
        'フレームの画像を作成'
        # フレームを読込み  
        frame = self.frame
    
        if frame is None:
            print("None")
        
        if self.contour:
            self.det_con=self.imgproc.detect_height(frame)
            self.imgproc.out.write(self.det_con)
            self.imgproc.calc_height()
            self.imgproc.write_log() 
            
        #PILイメージに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
                
        # 指定サイズに合わせて画像をリサイズ

        # 拡大率を計算
        ratio_x = size[0] / pil_image.width
        ratio_y = size[1] / pil_image.height

        if ratio_x < ratio_y:
            ratio = ratio_x
        else:
            ratio = ratio_y

        # リサイズ
        self.image = pil_image.resize(
            (
                int(ratio * pil_image.width),
                int(ratio * pil_image.height)
            )
        )
        t2 = time.time()
        #print(f"経過時間：{t2-t1}")

    def get_image(self):
        'Tkinter画像オブジェクトを取得する'

        if self.image is not None:
            # Tkinter画像オブジェクトに変換
            self.image_tk = ImageTk.PhotoImage(self.image)
        return self.image_tk

    def get_fps(self):
        '動画のFPSを取得する'

        if self.video is None:
            return None
        return self.video.get(cv2.CAP_PROP_FPS)
        
    def detect_contour(self):
        self.contour = not self.contour

    def start_timer(self):
        self.start=time.time()
    
    def calc_time(self):
        self.now=time.time()-self.start

    def get_frame(self):
        frame = self.frame
        cv2.imwrite('screenshot.jpg',frame)

class View(threading.Thread):

    def __init__(self, imgproc, app, model):
        threading.Thread.__init__(self)
        self.master = app
        self.model = model
        self.imgproc = imgproc

        # アプリ内のウィジェットを作成
        self.create_canvas()
        self.create_buttons()
        self.create_txtbox()
        self.create_spinbox()
        self.create_menu()
        
        #閾値の初期値設定
        self.imgproc.initial_HSV()
    
    ############################################
    #メニュー###################################
    def create_menu(self):
        #メニューバーの作成
        menubar=tkinter.Menu(self.master)
        self.master.config(menu=menubar)
        setting=tkinter.Menu(menubar,tearoff=0)
        setting.add_command(label="環境設定",command=self.create_setting_window)
        menubar.add_cascade(label='設定',menu=setting)
    
    
    def create_setting_window(self):
        setting_window=tkinter.Toplevel(self.master)
        setting_window.geometry('300x100')
        setting_window.title("環境設定")
        #画面上限下限のメジャーのメモリ設定
        #ボタン
        button_frame=tkinter.Frame(setting_window)
        button_frame.grid()
        screen_size_set=tkinter.Button(button_frame,text="設定",command=self.btn_click)
        screen_size_set.grid(row=2,column=4,columnspan=2)
        #上限テキストボックス
        label=tkinter.Label( button_frame,text="最大")
        label.grid(row=1,column=0,columnspan=2)
        self.maxheight=tkinter.Entry( button_frame,width=5)
        self.maxheight.grid(row=2,column=0,columnspan=2)
        #下限テキストボックス
        label=tkinter.Label( button_frame,text="最小")
        label.grid(row=1,column=2,columnspan=2)
        self.minheight=tkinter.Entry(button_frame,width=5)
        self.minheight.grid(row=2,column=2,columnspan=2)

        #shot102通信設定
        label=tkinter.Label( button_frame,text="baudrate")
        label.grid(row=3,column=0,columnspan=2)
        self.maxheight=tkinter.Entry( button_frame,width=5)
        self.maxheight.grid(row=4,column=0,columnspan=2)

        label=tkinter.Label( button_frame,text="waittime")
        label.grid(row=4,column=1,columnspan=2)
        self.maxheight=tkinter.Entry( button_frame,width=5)
        self.maxheight.grid(row=5,column=2,columnspan=2)

        label=tkinter.Label( button_frame,text="COM number")
        label.grid(row=6,column=2,columnspan=2)
        self.maxheight=tkinter.Entry( button_frame,width=5)
        self.maxheight.grid(row=7,column=3,columnspan=2)

    def btn_click(self):
        maxheight=float(self.maxheight.get())
        minheight=float(self.minheight.get())

        self.pxsize=str((maxheight-minheight)/480)

        path='./env_vle.txt'
        #if os.path.exists('env_vle.txt') == False:
            # with open(path,mode='w') as f:
            #    f.write('0.2333')
        with open(path,mode='w') as f:
                f.write(self.pxsize)


    
    def create_canvas(self):
        # キャンバスのサイズ
        canvas_width = 400
        canvas_height = 300

        # キャンバスとボタンを配置するフレームの作成と配置
        self.main_frame = tkinter.Frame(self.master,
                                        bg = "gray15")
        self.main_frame.pack()

        # キャンバスを配置するフレームの作成と配置
        self.canvas_frame = tkinter.Frame(self.main_frame,
                                        bg = "gray15")
        self.canvas_frame.grid(column = 1, row = 1)

        # ユーザ操作用フレームの作成と配置
        self.operation_frame = tkinter.Frame(self.main_frame,
                                            bg = "gray15")
        self.operation_frame.grid(column = 2, row = 2)
        
        #log出力用テキストボックスの作成と配置 
        self.logtxt_frame = tkinter.Frame(self.main_frame,
                                        bg = "gray15")
        self.logtxt_frame.grid(column = 2, row = 1)
        
        #閾値入力用のフレームの作成と配置
        self.InputThreshold_frame = tkinter.Frame(self.main_frame,bg = "gray15")
        self.InputThreshold_frame.grid(column = 1,row = 2,sticky = tkinter.W,padx = 80,pady = 10)

        # 動画表示用キャンバスの作成と配置
        self.canvas = tkinter.Canvas(
            self.canvas_frame,
            width = canvas_width,
            height = canvas_height,
            bg = "gray15",
        )       
        self.canvas.pack(padx=5,pady=5)
        
        #ログテキスト用キャンバスの作成と設置
        self.log_text_canvas = tkinter.Canvas(
            self.logtxt_frame,
            bg = "gray15"   
            )
        self.log_text_canvas.pack(padx = (0,10),pady = 5)
        
        #閾値入力ボックス用キャンバスの作成と配置
        self.InputThreshold_set_canvas = tkinter.Canvas(
            self.InputThreshold_frame,
            bg = "gray15"         
            )
        self.InputThreshold_set_canvas.grid(column = 1,row = 1,padx = 0,pady = 0)
        
    def create_buttons(self):
        'アプリ内にボタンを作成・配置する'
    
        #webカメラ接続ボタンの作成と設置
        self.webcameratxt = tkinter.StringVar()
        self.txt = "webカメラ　接続"
        self.webcameratxt.set(self.txt)
        
        self.webCamera_button = tkinter.Button(
            self.operation_frame,
            textvariable = self.webcameratxt,
            width = 30,
            height = 2
        )
        self.webCamera_button.pack(pady = (0,15),fill = tkinter.BOTH)
        
        # 輪郭検出ボタンの作成と配置 
        self.contour_button_txt = tkinter.StringVar()
        self.con_txt = "輪郭検出　開始"
        self.contour_button_txt.set(self.con_txt)
        
        self.contour_button = tkinter.Button(
            self.operation_frame,
            textvariable = self.contour_button_txt,
            width = 30,
            height = 2
        )
        self.contour_button.pack(pady = (0,15),fill = tkinter.BOTH)

        #　静止画撮影ボタンの作成と配置
        self.screenshot_button_txt = tkinter.StringVar()
        self.screenshot_txt = "スクリーンショット"
        self.screenshot_button_txt.set(self.screenshot_txt)

        self.screenshot_button = tkinter.Button(
            self.operation_frame,
            textvariable = self.screenshot_button_txt,
            width = 30,
            height = 2,
        )
        self.screenshot_button.pack(pady = (0,15),fill = tkinter.BOTH)
        
        #霧検出閾値入力ボタンの作成・設置
        self.initialset_button = tkinter.Button(
            self.InputThreshold_set_canvas,
            text = "設定"
        )
        self.initialset_button.grid(row=3,column=3,padx=5,pady=30)
        
    def create_txtbox(self):
        'テキストボックスの作成・設置'
        #tkinter.Label(self.log_text_canvas,text="log出力").pack(padx=(0,150),pady=(3,3),side=tkinter.TOP)
        self.log_text = tkinter.Text(self.log_text_canvas,
                                width = 37,
                                height = 23,
                                bd = 3, 
                                relief = "groove"                                 
                                )

        self.scrollbar = tkinter.Scrollbar(self.log_text_canvas,command = self.log_text.yview)
        self.scrollbar.pack(side = "right",fill = "y",padx = (0,20),pady = (10,10))
        self.log_text['yscrollcommand'] = self.scrollbar.set
        self.log_text.pack(side = tkinter.LEFT,padx = (10,0),pady = (10,10))        
        
        #最大噴霧高さ表示ラベル
        tkinter.Label(self.logtxt_frame,
                    text = "最大噴霧高さ",
                    fg = 'white',
                    bg = 'gray15').pack(side = tkinter.LEFT,padx = (10,0),pady = (10,10))

        #最大噴霧高表示テキストボックス
        self.max_show_text = tkinter.Text(self.logtxt_frame,
                                    width = 8,
                                    height = 1,
                                    bd = 1, 
                                    )
        self.max_show_text.pack(side = tkinter.LEFT,padx = (10,0),pady = (10,10))
        self.max_show_text.insert(1.0,'0')

        #平均噴霧高さ表示ラベル
        tkinter.Label(self.logtxt_frame,
        text = '平均噴霧高さ',
        fg = 'white',
        bg = 'gray15').pack(side = tkinter.LEFT,padx = (10,0),pady = (10,10))

        #平均噴霧高さ表示テキストボックス
        self.ave_show_text = tkinter.Text(self.logtxt_frame,
        width = 8,
        height = 1,
        bd = 1,
        )
        self.ave_show_text.pack(side = tkinter.LEFT,padx = (10,0),pady = (10,10))
        self.ave_show_text.insert(1.0,'0')

        #3min計測用チェックボックスラベル
        #tkinter.Label(self.logtxt_frame,
        #text='3分計測',
        #fg='white',
        #bg='gray').pack(side=tkinter.LEFT,padx=(10,0),pady=(10,10))

        #3min計測用チェックボックス設置
        self.chkValue = tkinter.IntVar() 
        self.chkValue.set(1)
        self.time_check = tkinter.Checkbutton(self.operation_frame,text = '3分で終了',var = self.chkValue)
        self.time_check.pack(side = tkinter.LEFT,padx = (10,0),pady = (10,10))

        #測りの高さをオフセットする用チェックボックス設置
        self.offsetValue = tkinter.IntVar()
        self.offsetValue.set(1)
        self.offset_check=tkinter.Checkbutton(self.operation_frame,text = '測りあり',var = self.offsetValue)
        self.offset_check.pack(side = tkinter.LEFT,padx = (10,0),pady = (10,10),ipadx = 1)

    def changetxt(self):
        'webカメラボタンの表示変更'
        if self.txt == "webカメラ　接続":
            self.txt = "webカメラ　切断"
            self.webcameratxt.set(self.txt)
        else:
            self.txt = "webカメラ　接続"
            self.webcameratxt.set("webカメラ　接続")
                
    def change_contour_button_txt(self):
        '輪郭検出ボタンの表示変更'
        
        if self.con_txt == "輪郭検出　開始":
            self.con_txt = "輪郭検出　停止"
            self.contour_button_txt.set(self.con_txt)
        else:
            self.con_txt = "輪郭検出　開始"
            self.contour_button_txt.set(self.con_txt)

    def create_spinbox(self):
        '輪郭検出閾値の入力用ボックスの作成・設置'
        
        self.lower_color_frame = tkinter.Frame(self.InputThreshold_set_canvas,bg = 'gray15',padx = 0,pady = 0)
        self.lower_color_frame.grid(row = 3,column = 1,padx = (10,0),pady = 0)
        self.upper_color_frame = tkinter.Frame(self.InputThreshold_set_canvas,bg = 'gray15')
        self.upper_color_frame.grid(row = 3,column = 2,padx = 0)
        
        #霧検出閾値入力スピンボックスの作成・配置
        Spinwidth=3
        tkinter.Label(self.InputThreshold_set_canvas,
                    text = "霧検出閾値設定",
                    fg = 'white',
                    bg = 'gray15').grid(row = 1,column = 1,ipadx = 0,pady = (10,0))
        
        #霧検出下限閾値
        tkinter.Label(self.InputThreshold_set_canvas,
                    text = "下限閾値",
                    fg = 'white',
                    bg = 'gray15').grid(row = 2,column = 1,ipadx = 0,ipady = 0)   
        
        #下限閾値B
        tkinter.Label(self.lower_color_frame,text = "H",
                    fg = 'white',
                    bg = 'gray15').pack(pady = 0,side = tkinter.LEFT)  
                    
        low_numB = tkinter.StringVar()
        low_numB.set(self.imgproc.lowerH)
        self.lower_colorB = tkinter.Spinbox(self.lower_color_frame,
                                        textvariable = low_numB,
                                        from_= 0,
                                        to = 255,
                                        increment = 1,
                                        width = Spinwidth
                                        )
        self.lower_colorB.pack(ipadx = 0,ipady = 0,padx = 0,pady = 0,side = tkinter.LEFT)
        
        #下限閾値G
        tkinter.Label(self.lower_color_frame,text = "S",
                    fg = 'white',
                    bg = 'gray15').pack(padx = 0,pady = 0,side = tkinter.LEFT)
        low_numG = tkinter.StringVar()
        low_numG.set(self.imgproc.lowerS)
        self.lower_colorG = tkinter.Spinbox(self.lower_color_frame,
                                        from_ = 0,
                                        textvariable = low_numG,
                                        to = 255,
                                        increment = 1,
                                        width = Spinwidth
                                        )
        self.lower_colorG.pack(padx = 0,pady = 0,side = tkinter.LEFT)
        
        #下限閾値R
        tkinter.Label(self.lower_color_frame,text = "V",
                    fg = 'white',
                    bg = 'gray15').pack(padx = 0,pady = 0,side = tkinter.LEFT)
        low_numR = tkinter.StringVar()
        low_numR.set(self.imgproc.lowerV)

        self.lower_colorR=tkinter.Spinbox(self.lower_color_frame,
                                        textvariable=low_numR,
                                        from_= 0,
                                        to = 255,
                                        increment = 1,
                                        width = Spinwidth
                                        )

        self.lower_colorR.pack(padx = 0,pady = 0,side = tkinter.LEFT)
            
        #霧検出上限閾値
        tkinter.Label(self.InputThreshold_set_canvas,
                    text = "上限閾値",
                    fg = 'white',
                    bg = 'gray15').grid(row = 2,column = 2,padx = 0,pady = 0)
        
        #上限閾値B
        tkinter.Label(self.upper_color_frame,text = "H",
                    fg = 'white',
                    bg = 'gray15').pack(padx = 0,pady = 0,side = tkinter.LEFT)
        upp_numB = tkinter.StringVar()
        upp_numB.set(self.imgproc.upperH)
        self.upper_colorB = tkinter.Spinbox(self.upper_color_frame,
                                        textvariable = upp_numB,
                                        from_= 0,
                                        to = 255,
                                        increment = 1,
                                        width = Spinwidth,
                                        )
        self.upper_colorB.pack(padx = 0,pady = 0,side = tkinter.LEFT)
        
        #上限閾値G
        tkinter.Label(self.upper_color_frame,text="S", fg='white',bg='gray15').pack(padx = 0,pady = 0,side = tkinter.LEFT)
        upp_numG = tkinter.StringVar()
        upp_numG.set(self.imgproc.upperS)
        self.upper_colorG = tkinter.Spinbox(self.upper_color_frame,
                                        textvariable = upp_numG,
                                        from_= 0,
                                        to = 255,
                                        increment = 1,
                                        width=Spinwidth
                                        )
        self.upper_colorG.pack(padx = 0,pady = 0,side = tkinter.LEFT)

        #上限閾値R
        tkinter.Label(self.upper_color_frame,text = "V",fg='white',bg='gray15').pack(padx = 0,pady =0 ,side = tkinter.LEFT)
        upp_numR = tkinter.StringVar()
        upp_numR.set(self.imgproc.upperV)
        self.upper_colorR=tkinter.Spinbox(self.upper_color_frame,
                                        textvariable=upp_numR,
                                        from_= 0,
                                        to = 255,
                                        increment = 1,
                                        width = Spinwidth
                                        )
        self.upper_colorR.pack(padx = 0,pady = 0,side = tkinter.LEFT)
        
    def draw_image(self):
        '画像をキャンバスに描画'
        image = self.model.get_image()
        
        if image is not None:
            # キャンバス上の画像の左上座標を決定
            sx = (self.canvas.winfo_width() - image.width()) // 2
            sy = (self.canvas.winfo_height() - image.height()) // 2
            print('sx',sx,'sy',self.canvas.winfo_height())
            # キャンバスに描画済みの画像を削除
            #objs = self.canvas.find_withtag("image")
            self.canvas.delete("image")
            #for obj in objs:
                #self.canvas.delete(obj)
                    
            # 画像をキャンバスの真ん中に描画
            self.canvas.create_image(
                sx, sy,
                image = image,
                anchor = tkinter.NW,
                tag = "image"
            )
            self.delete_draw_reticle()
            self.draw_reticle()

    def draw_play_button(self):
        '再生ボタンを描画'
        # キャンバスのサイズ取得
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        # 円の直径を決定
        if width > height:
            diameter = height
        else:
            diameter = width

        # 端からの距離を計算
        distance = diameter / 10

        # 円の線の太さを計算
        thickness = distance

        # 円の描画位置を決定
        sx = (width - diameter) // 2 + distance
        sy = (height - diameter) // 2 + distance
        ex = width - (width - diameter) // 2 - distance
        ey = height - (height - diameter) // 2 - distance

        # 丸を描画
        self.canvas.create_oval(
            sx, sy, 
            ex, ey,
            outline = "white",
            width = thickness,
            tag = "oval"
        )

        # 頂点座標を計算
        x1 = sx + distance * 3
        y1 = sy + distance * 2
        x2 = sx + distance * 3
        y2 = ey - distance * 2
        x3 = ex - distance * 2
        y3 = height // 2

        # 三角を描画
        self.canvas.create_polygon(
            x1, y1,
            x2, y2,
            x3, y3,
            fill = "white",
            tag = "triangle"
        )
    #webカメラからの取得画像に目盛りの表示
    def draw_reticle(self):
        # キャンバスのサイズ取得
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        # 円の直径を決定
        if width > height:
            diameter = height
        else:
            diameter = width

        sx=width/2
        sy=height/2
        x_large=sx+75
        x_small=sx+65
        y_large=sy-73
        y_small=sy-18
        #print('180cm',y_large)
        #print('160cm',y_small)
     
        #カメラ設置位置基準レティクルの作成
        self.canvas.create_line(x_large,y_large,x_large,y_small,fill = "red",tag = "reticle",)
        self.canvas.create_line(x_large,y_large,x_small,y_large,fill = "red",tag = "reticle",)
        self.canvas.create_line(x_large,y_small,x_small,y_small,fill = "red",tag = "reticle",)
        #レティクル高さ表記
        self.canvas.create_text(x_large+20,y_large,text="180cm",font=("",10,),tag="reticle",)
        self.canvas.create_text(x_large+20,y_small,text="160cm",font=("",10,),tag="reticle",)

    def delete_play_button(self):
        self.canvas.delete("oval")
        self.canvas.delete("triangle")

    def delete_draw_image(self):
        # キャンバスに描画済みの画像を削除
        self.canvas.delete("image")
        #self.canvas.delete("reticle")

    def delete_draw_reticle(self):
        self.canvas.delete("reticle")

class Controller():

    def __init__(self, imgproc, app, model, view):
        
        self.master = app
        self.model = model
        self.view = view
        self.imgproc = imgproc       
        self.cntframe = 0
        self.timeold = time.time()
        self.timelist = []
        
        # 動画再生中かどうかの管理
        self.playing = False
        #輪郭検出中かどうかの管理
        self.contours = False
        #webカメラ使用中かどうかの管理
        self.webcamera = False
        #動画選択中かどうかの管理
        self.video = False
        #キャンバス上に表示されるオブジェクトの設定
        self.canvasImg = False
        # フレーム進行する間隔（前まで60)
        self.frame_timer = 0
        # 描画する間隔90
        self.draw_timer = 120
    
        self.set_events()

    def set_events(self):
        '受け付けるイベントを設定する'
        # キャンバス上のマウス押し下げ開始イベント受付
        self.view.canvas.bind("<Button>",self.button_press)
        # 輪郭検出開始ボタン押し下げイベント受付
        self.view.contour_button['command'] = self.push_contour_button
        #webカメラボタン押し下げイベント受付
        self.view.webCamera_button['command'] = self.push_webCamera_button
        #スクリーンショットボタン押し下げイベント受付
        self.view.screenshot_button['command'] = self.push_screenshot_button
        #閾値設定ボタン押し下げイベント受付
        self.view.initialset_button['command'] = self.push_initialset_button
        
    def draw(self):       
        '一定間隔で画像等を描画'
        start = time.time()
        # 再度タイマー設定
        self.master.after(self.draw_timer, self.draw)
        self.cntframe = self.cntframe+1
        
        # 動画再生中の場合
        if self.playing:
            
            # フレームの画像を作成
            self.model.create_image(
                (self.view.canvas.winfo_width(),self.view.canvas.winfo_height())
            )
            # 動画１フレーム分をキャンバスに描画
            self.view.draw_image()
            if self.model.contour != True:
                #self.view.log_text.insert(1.0,'None')
            #self.view.log_text.insert(1.0,'\n')
                pass
            else:                
                #logテキストボックスにlogの書き込み
                log = self.imgproc.write_log()
                #self.view.log.text.insert(1.0,'\n')
                self.view.log_text.insert(1.0,log)
                self.view.log_text.insert(1.0,'\n')
                #最大噴霧高さテキストボックスに最大噴霧高の書き込み
                self.view.max_show_text.delete(1.0,tkinter.END)
                self.view.max_show_text.insert(1.0,self.imgproc.maxNum)
                #平均噴霧高さテキストボックスに平均噴霧高さの書き込み
                self.view.ave_show_text.delete(1.0,tkinter.END)
                self.view.ave_show_text.insert(1.0,self.imgproc.ave_realheight)
                timenow = time.time()
                proctime1 = timenow-self.timeold
                #print(proctime)
                self.imgproc.proctime(proctime1)
                self.timeold = timenow
                
    def frame(self):
        '一定間隔でフレームを進める'
        # 再度タイマー設定
        self.master.after(self.frame_timer, self.frame)
        # 動画再生中の場合
        if self.playing == False:
            return
        # 動画を１フレーム進める
        ret = self.model.advance_frame()
        # フレームが進められない場合
        if not ret:
            # フレームを最初に戻す
            self.playing=False              
            #self.model.advance_frame()
            self.model.reverse_video()
            self.model.advance_frame()
    
    def push_initialset_button(self): 
        #HSV下限値設定
        self.imgproc.lowerH = int(self.view.lower_colorB.get())
        self.imgproc.lowerS = int(self.view.lower_colorG.get())
        self.imgproc.lowerV = int(self.view.lower_colorR.get())
        
        #HSV上限値設定
        self.imgproc.upperH = int(self.view.upper_colorB.get())
        self.imgproc.upperS = int(self.view.upper_colorG.get())
        self.imgproc.upperV = int(self.view.upper_colorR.get())
        
    def create_videoimage(self):      
        if len(self.file_path) == 0:
            return
        
        # 動画オブジェクト生成
        self.model.create_video(self.file_path)
        if self.model.video_open == True:
            # 最初のフレームを表示
            self.model.advance_frame()
            self.model.create_image(
                (
                    self.view.canvas.winfo_width(),
                    self.view.canvas.winfo_height()
                )
            )
            #print(self.view.canvas.winfo_width())
            self.model.reverse_video()
            self.view.draw_image()
            # 再生ボタンの表示
            self.view.delete_play_button()
            self.view.draw_play_button()
            #self.view.draw_reticle()
            #FPSに合わせてフレームを進める間隔を決定
            fps = self.model.get_fps()
            #print(fps)
            self.frame_timer = int(1 / fps * 1000 + 0.5)
            #print('advance frame ')
            #フレーム進行用のタイマースタート
            self.master.after(self.frame_timer, self.frame)
            #画像の描画用のタイマーセット
            self.master.after(self.draw_timer, self.draw)
            
    def button_press(self,event):
        'マウスボタン押された時の処理'
        # 動画の再生/停止を切り替える
        if not self.playing:
            self.playing = True
            # 再生ボタンの削除
            self.view.delete_play_button()
            
        else:
            self.playing = False
            # 再生ボタンの描画
            self.view.delete_draw_reticle()
            self.view.draw_play_button()
        print(self.view.chkValue.get())

    def push_webCamera_button(self):        
        
        if self.view.txt == "webカメラ　接続":          
            self.canvasImg = True
            self.file_path = 'web'
            self.webcamera = True         
            self.create_videoimage()
            self.view.delete_draw_reticle()
        else:       
            self.model.release_video()
            self.view.delete_draw_image()
            self.view.delete_draw_reticle()
            self.view.delete_play_button()
            
        if self.model.video_open == True: 
            #self.webcamera = False              
            self.view.changetxt()
        #print(self.webcamera)    
        
    def push_contour_button(self):
        '輪郭検出ボタン押されたときの処理'   
        if self.webcamera != True :
            return
        if self.model.video_open != True:  
            return
        self.model.detect_contour()
                
        if self.view.con_txt == "輪郭検出　開始":

            self.imgproc.create_img_list()
            self.view.max_show_text.insert(1.0,0)
            self.view.ave_show_text.insert(1.0,0)
            if self.view.offsetValue.get() == 1:
                self.imgproc.offset = -1.5
            else:
                self.imgproc.offset = 0
            self.imgproc.rec_video(self.model.fps,self.model.width,self.model.height)
            if self.view.chkValue.get() == 1:
            
                self.master.after(180000, self.stop)
                #self.stop_timer=self.master.after(5000,self.push_contour_button)
                if self.view.contour_button['state'] == "normal":
                    self.view.contour_button['state'] = "disabled"
            self.control_shot102()
        else:
            #print(self.view.con_txt)
            self.imgproc.rec_fin()
            #self.master.after.cancel(self.stop_timer)
        self.view.change_contour_button_txt()

    def push_screenshot_button(self):
        'スクリーンショットボタン押したときの処理'
        self.model.get_frame()
        self.control_shot102()
        
    def stop(self):
        self.model.detect_contour()
        self.imgproc.rec_fin()
        self.view.change_contour_button_txt()
        if self.view.contour_button['state'] == "disabled":
                self.view.contour_button['state'] = "normal"

    def push_flip_button(self):
        self.model.set_flip()
        self.view.change_contour_button_txt()

    #shot102の制御
    def control_shot102(self):

        self.detecting_contour_flag =False
        self.move_direction = True

        self.baudrate=38400
        self.write_waittime=0.05
        stage = shot_102("COM18",pulse_per_unit=(1000,1000),baudrate=38400,write_waittime=0.05)
        if self.detecting_contour_flag == False:
            pass

        self.master.after(self.timer,self.control_shot102)
        sleep(0.1)

        if self.move_direction == True:
            stage.set_cmd_relative_move_unit(1,-3000)
            stage.start_move()
            print("a")
            self.move_direction = False
        else:
            stage.set_cmd_relative_move_pulse(1,+3000)
            stage.start_move()
            self.move_direction = True


app = tkinter.Tk()
app.title("ミスト検出アプリver1.02")

imgproc = Image_Proc()
imgproc.start()

model = Model(imgproc)
model.start()

view = View(imgproc,app, model)
view.start()

controller = Controller(imgproc, app, model, view)

app.mainloop()
