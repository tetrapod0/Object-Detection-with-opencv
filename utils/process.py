from gui.main import colors
from utils.logger import logger
from utils.text import *
from utils import tool
from collections import defaultdict
from PIL import ImageFont, ImageDraw, Image, ImageTk
from threading import Thread
from functools import reduce
import numpy as np
import tkinter as tk
import traceback
import time
import cv2
import os
import re

class Stopper():
    def __init__(self):
        self.stop_signal = False

#####################################################
def image_eater(self, image_frame_list, image_label_list):
    thread_cycle = 0.05
    origin_image_list = []
    last_winfo_list = [(frame.winfo_height(), frame.winfo_width()) for frame in image_frame_list]
    
    try:
        while not self.stop_signal:
            time.sleep(thread_cycle)

            # 현재 프레임 크기 가져오기
            current_winfo_list = [(frame.winfo_height(), frame.winfo_width()) for frame in image_frame_list]

            # GUI 이미지 업데이트 조건 검사
            if current_winfo_list == last_winfo_list and self.image_Q.empty(): continue
            if current_winfo_list != last_winfo_list: last_winfo_list = current_winfo_list
            if not self.image_Q.empty(): origin_image_list = self.image_Q.get() # BGR
            if not origin_image_list: continue
            assert len(origin_image_list) == len(image_frame_list)

            # 이미지 변형
            # imgtk_list = [None, None, None]
            # self.current_origin_image = origin_image_list[0]
            
            for i in range(len(origin_image_list)):
                frame, img, label = image_frame_list[i], origin_image_list[i], image_label_list[i]
                winfo = frame.winfo_height(), frame.winfo_width()
                if winfo == (1,1): continue

                # GUI 이미지 업데이트
                if img is None:
                    # label.configure(image=None)
                    # label.image = None
                    # imgtk_list[i] = None
                    label.pack_forget()
                else:
                    img, _ = fit_img(img[:,:,::-1], winfo)
                    imgtk = ImageTk.PhotoImage(Image.fromarray(img), master=self)
                    label.configure(image=imgtk)
                    label.image = imgtk
                    # imgtk_list[i] = imgtk
                    label.pack(expand=True, fill="both")
                
    
    except Exception as e:
        logger.error("an error in [image_eater]")
        logger.error(traceback.format_exc())
        self.stop_signal = True
        
    # 이미지 없애기
    for label in image_label_list:
        # label.configure(image=None)
        # label.image = None
        label.pack_forget()

    self.current_origin_image = None # 파일 저장용
    
def fit_img(img, size, margin=15):
    wh, ww = size
    wh, ww = wh-margin, ww-margin
    h, w = img.shape[:2]
    magnf_value = min(wh/h, ww/w)
    new_img = cv2.resize(img, dsize=(0,0), fx=magnf_value, fy=magnf_value)
    return new_img, magnf_value

#####################################################
def data_eater(self):
    thread_cycle = 0.05
        
    while not self.stop_signal:
        time.sleep(thread_cycle)
        
        if self.data_Q.empty(): continue    
        name, isok, date = self.data_Q.get()
            
        # 해당 지시코드의 데이터 수정
        seletec_col = "OK" if isok else "NG"
        
        # 세부데이터 업데이트
        self.value_label1.configure(text=name if name else "미탐지 또는 새로운 품목") # 판독품목
        self.value_label2.configure(text=date if date else "") # 판독날짜
        
        # OK, NG
        if isok: self.ok_label.configure(text='OK', bg=colors[2], fg=colors[1])
        else: self.ok_label.configure(text='NG', bg=colors[4], fg=colors[1])

#######################################################################
def read(self):
    thread_cycle = 0.05
    
    try:
        while not self.stop_signal:
            time.sleep(thread_cycle)
            if self.raw_Q.empty(): continue

            # get image
            img = self.raw_Q.get()
            if img is None: self.analy_Q.put([None, None, None, None])
            
            # Detect Polys
            start_time = time.time()
            best_obj, dst_polys, crop_imgs = self.poly_detector.predict(img)
            end_time = time.time()
            logger.info(f"Detect Time : {end_time-start_time:.3f}")
            
            self.analy_Q.put([img, best_obj, dst_polys, crop_imgs])
            
    except Exception as e:
        logger.error("an error in [read]")
        logger.error(traceback.format_exc())
        self.stop_signal = True

#######################################################################
def analysis(self):
    thread_cycle =  0.05
    
    try:
        while not self.stop_signal:
            time.sleep(thread_cycle)
            if self.analy_Q.empty(): continue

            # poly 결과 받기
            img, best_obj, dst_polys, crop_imgs = self.analy_Q.get()
            
            #
            isok = True
            data = None
            
            # 미탐지한 경우
            if best_obj is None:
                isok = False
            # 날짜있는 제품인 경우
            elif "data" in best_obj.labels:
                i = best_obj.labels.index("data")
                data_img = crop_imgs[i]
                data, _ = self.ocr_engine(data_img, use_beam=True)
                
            name = best_obj.name if best_obj else None
            
            self.data_Q.put([name, isok, data])
            self.draw_Q.put([img, isok, data, best_obj, dst_polys, crop_imgs]) # recode 때문에 isok필요
        
    except Exception as e:
        logger.error("an error in [analysis]")
        logger.error(traceback.format_exc())
        self.stop_signal = True
        
#######################################################################
def draw(self):
    thread_cycle = 0.05
    
    # 색깔 초기화
    fc = lambda x,y:np.random.randint(x,y)
    colors = [(fc(50,255), fc(50,255), fc(0,150)) for _ in range(len(self.poly_detector.names))]
    color_dic = dict(zip(self.poly_detector.names, colors))
    font_cv = cv2.FONT_HERSHEY_SIMPLEX
    font_pil = ImageFont.truetype(FONT_PATH, 30)
    
    try:
        while not self.stop_signal:
            time.sleep(thread_cycle)
            if self.draw_Q.empty(): continue
            
            img, isok, data, best_obj, dst_polys, crop_imgs = self.draw_Q.get()

            # 미탐지인 경우
            if best_obj is None:
                self.image_Q.put([img, None, None])
                continue
            
            #
            name = best_obj.name
            label2idx = dict(zip(best_obj.labels, range(len(best_obj.labels))))
            dst_polys = dst_polys.astype(np.int32)
            obj_img, date_img = None, None
            
            # 폴리곤 그리기
            color = color_dic[name]
            cv2.polylines(img, dst_polys, True, color, thickness=5)
            
            # 숫자 그리기
            for dst_poly in dst_polys:
                cv2.putText(img, "1", dst_poly[0], font_cv, fontScale=1, thickness=3, color=(255,0,255))
                cv2.putText(img, "2", dst_poly[1], font_cv, fontScale=1, thickness=3, color=(255,0,255))
                cv2.putText(img, "3", dst_poly[2], font_cv, fontScale=1, thickness=3, color=(255,0,255))
                cv2.putText(img, "4", dst_poly[3], font_cv, fontScale=1, thickness=3, color=(255,0,255))
            
            # ndarr -> pil
            img_pil = Image.fromarray(img)
            img_draw = ImageDraw.Draw(img_pil)
            
            # name 그리기
            i = label2idx["object"]
            obj_img = crop_imgs[i]
            x, y = dst_polys[i, 0, 0], dst_polys[i, 0, 1]-40
            img_draw.text((x,y), name, font=font_pil, fill=(*color, 0))
            
            # draw anno2
            if "data" in label2idx:
                i = label2idx["data"]
                date_img = crop_imgs[i]
                x, y = dst_polys[i, 0, 0], dst_polys[i, 0, 1]-40
                img_draw.text((x,y), data, font=font_pil, fill=(*color, 0))
            
            # pil -> ndarr
            img = np.array(img_pil)
            
            self.image_Q.put([img, obj_img, date_img])
        
    except Exception as e:
        logger.error("an error in [draw]")
        logger.error(traceback.format_exc())
        self.stop_signal = True

#######################################################################

def find_poly_thread(self):
    thread_cycle = 0.05
    brightness = 60
    kernel = np.ones((3,3))
    temp_poly = np.array([[50,50], [100,50], [100,100], [50,100]])
    
    try:
        while not self.stop_signal:
            time.sleep(thread_cycle)
            if self.raw_Q.empty(): continue

            # get image
            img = self.raw_Q.get()
            if img is None: continue
            
            # Detect Polys
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_mask = cv2.inRange(img_hsv, (0, 0, brightness), (360, 255, 255))
            img_mask = cv2.erode(img_mask, kernel, iterations=3)
            img_mask = cv2.dilate(img_mask, kernel, iterations=3)
            polys = tool.find_polys_in_img(img_mask)
            
            self.poly1 = temp_poly.copy() if polys is None else tool.poly2clock(polys[0])
            self.poly2 = temp_poly.copy() if self.poly2 is not None else None
            self.image1 = img
            self.image_update()
            
            # 자동촬영 스위치 끄기
            # self.auto_stopper.stop_signal = True
            # self.bf_btn2.switch_on = False
            # self.bf_btn2.configure(bg="#393945", fg="#A6A6A6")
            
    except Exception as e:
        logger.error("an error in [find_poly_thread]")
        logger.error(traceback.format_exc())
        self.stop_signal = True
        










