from gui.addwin import configure, colors
from control.labelwin import LabelWindow

from utils import tool
from utils import process
from utils.text import *

import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter import messagebox as mb
from PIL import ImageTk, Image
from threading import Thread, Lock
from queue import Queue
import numpy as np
import time
import cv2
import os

class AddWindow(tk.Toplevel):
    def __init__(self, name, callback=None):
        super().__init__()
        self.title("Apply Window")
        self.focus() # 창 선택해줌
        self.grab_set() # 다른창 못건드림
        
        # window size
        self.geometry(f"{self.winfo_screenwidth()//5*4}x{self.winfo_screenheight()//5*4}")
        self.resizable(False, False)
        self.overrideredirect(False)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.win_factor = self.winfo_screenheight() / 1980
        
        self.name = name
        self.callback = callback
        
        # images
        self.image1 = None
        self.image2 = None
        self.image3 = None
        # polys
        self.poly1 = None
        self.poly2 = None
        # M
        self.M1 = None
        self.M2 = None
        
        self.auto_stopper = process.Stopper()
        
        # GUI 및 bind
        configure(self)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.set_bind()
        
        # 항상 실행
        self.stop_signal = False
        self.raw_Q = Queue()
        self.image_Q = Queue()
        image_frame_list = [self.tf1_frame, self.tf2_frame, self.tf3_frame, ]
        image_label_list = [self.tf1f_label, self.tf2f_label, self.tf3f_label, ]
        Thread(target=process.image_eater, args=(self, image_frame_list, image_label_list), daemon=True).start()
        Thread(target=process.find_poly_thread, args=(self,), daemon=True).start()
        
###########################################################################################        
    def trigger_btn(self):
        img = tool.get_url_img(IMG_URL)
        self.raw_Q.put(img)
    
###########################################################################################
    def stop(self):
        self.stop_signal = True
        time.sleep(0.1)
        self.destroy()
        self.callback(None)

    def on_closing(self):
        Thread(target=self.stop, args=(), daemon=True).start()

###########################################################################################
    def open_file(self):
        # 이미지 가져오기, 포커스 고정
        top = tk.Toplevel(self)
        top.grab_set() # 포커스 고정
        top.withdraw() # 숨기기
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("image files", "*.png"),
                                                         ("image files", "*.jpg")))
        top.grab_release()
        # top.deiconify()
        top.destroy()
        self.grab_set() # 다시 서브창 포커스
        
        if not filename: return
        try: origin_img_arr = tool.imread(filename)
        except:
            mb.showwarning(title="", message="올바른 파일이 아닙니다.")
            return
        
        # 이미지 적용
        self.raw_Q.put(origin_img_arr)
        
        # self.bf_btn4.configure(bg="#393945", fg="#A6A6A6")
        
###########################################################################################
    def submit(self):
        if self.image1 is None or self.image2 is None:
            mb.showwarning(title="", message="촬영되지 않았습니다.")
            return
        
        # answer = mb.askquestion("등록하기", f"해당 품목을 등록하시겠습니까?")
        # if answer == "no": return
            
        
        obj_poly = self.poly1
        date_poly = self.poly2
        img = self.image1
        
        # 날짜 poly 변환
        if date_poly is not None:
            obj_poly, date_poly = obj_poly.astype(np.float32), date_poly.astype(np.float32)
            h, w = self.image2.shape[:2]
            pos = np.float32([[0,0], [w,0], [w,h], [0,h]])
            M = cv2.getPerspectiveTransform(pos, obj_poly)
            
            date_poly = cv2.perspectiveTransform(date_poly.reshape(-1,1,2), M).reshape(-1,2)
        
        # json 저장
        path = os.path.join(LABEL_PATH, f"{self.name}.json")
        if date_poly is not None:
            tool.poly2json(path, ["object", "data"], [obj_poly, date_poly])
        else:
            tool.poly2json(path, ["object"], [obj_poly])

        # img 저장
        path = os.path.join(IMAGE_PATH, f"{self.name}.png")
        tool.imwrite(path, img)

        # 메인창 GUI업데이트
        self.callback(self.name)
        
        self.on_closing()
        # self.destroy()

###########################################################################################
    def image_update(self):
        # step1
        image1 = self.image1.copy() if self.image1 is not None else None
        poly1 = self.poly1.astype(np.int32) if self.poly1 is not None else None
        if image1 is not None and self.poly1 is not None:
            self.image2, _ = tool.get_crop_img_and_M(image1, self.poly1)
        else: self.image2 = None
        cv2.polylines(image1, [poly1], True, (255,255,0), thickness=5)
        # step2
        image2 = self.image2.copy() if self.image2 is not None else None
        poly2 = self.poly2.astype(np.int32) if self.poly2 is not None else None
        if image2 is not None and self.poly2 is not None:
            self.image3, _ = tool.get_crop_img_and_M(image2, self.poly2)
        else: self.image3 = None
        cv2.polylines(image2, [poly2], True, (255,255,0), thickness=5)
        
        self.image_Q.put([image1, image2, self.image3])
    
    def apply_poly(self, poly, n):
        if n==0: self.poly1 = poly
        else: self.poly2 = poly
        
        self.image_update()
        
    def reset(self):
        self.image1, self.image2, self.image3 = None, None, None
        self.poly1, self.poly2 = None, None
        self.image_Q.put([None, None, None])
        
    def date_on_off(self, switch_on):
        if switch_on: self.poly2 = np.array([[0,0], [50,0], [50,50], [0,50]])
        else: self.poly2 = None
        
        self.image_update()
        
###########################################################################################
    def set_bind(self):
        def on_push(event):
            if not hasattr(event.widget, 'do_button'): return
            event.widget.configure(bg=colors[1], fg=colors[2])
            event.widget.do_button()

        def on_leave(event):
            if not hasattr(event.widget, 'do_button'): return
            event.widget.configure(bg=colors[1], fg=colors[2])
            
        def switch(event):
            if not hasattr(event.widget, 'switch_on'): return
            if not hasattr(event.widget, 'do_switch_func'): return
        
            # 스위치
            event.widget.switch_on ^= True
            
            # 켜져 있다면
            if event.widget.switch_on:
                event.widget.configure(bg=colors[0], fg=colors[2])
                event.widget.do_switch_func(event.widget.switch_on)
            else:
                event.widget.configure(bg=colors[1], fg=colors[2])
                event.widget.do_switch_func(event.widget.switch_on)

        # 클릭모션 부여
        btn_list = [self.bf_btn3, self.bf_btn4, self.bf_btn5, self.bf_btn6, ]
        for btn in btn_list:
            btn.bind("<Button-1>", on_push)
            btn.bind("<ButtonRelease-1>", on_leave)
        
        # 스위칭 모션 부여
        self.bf_btn1.bind("<Button-1>", switch)
        # self.bf_btn2.bind("<Button-1>", switch)
        
        # 날짜유무 스위칭 기능
        self.bf_btn1.switch_on = False
        self.bf_btn1.do_switch_func = self.date_on_off
        # self.bf_btn1.do_switch_func(self.bf_btn1.switch_on)
        
        # 자동촬영 스위칭 기능
        # self.bf_btn2.switch_on = False
        # self.bf_btn2.do_switch_func = self.auto_cam
        # self.bf_btn2.do_switch_func(self.bf_btn2.switch_on)
        
        # 수동촬영 버튼 기능
        self.bf_btn3.do_button = self.trigger_btn
        # 파일열기 버튼 기능
        self.bf_btn4.do_button = self.open_file
        # 초기화 버튼 기능
        self.bf_btn5.do_button = self.reset
        # 등록완료 버튼 기능
        self.bf_btn6.do_button = self.submit
        
        def rotate_poly(n):
            # if self.bf_btn2.switch_on:
            #     mb.showwarning(title="", message="자동촬영을 종료해주세요.")
            #     return
            
            if n == 0 and self.poly1 is not None:
                self.poly1 = self.poly1[[3,0,1,2]]
                self.image_update()
            elif n == 1 and self.poly1 is not None:
                self.poly2 = self.poly2[[3,0,1,2]]
                self.image_update()
            
        def edit_points(n):
            # if self.bf_btn2.switch_on:
            #     mb.showwarning(title="", message="자동촬영을 종료해주세요.")
            #     return
            
            init_poly = self.poly1 if n==0 else self.poly2
            init_image = self.image1 if n==0 else self.image2
            if init_poly is not None:
                LabelWindow(init_image, points=init_poly, callback=lambda poly:self.apply_poly(poly,n), )
        
        # 좌표수정1
        self.tf1_btn1["command"] = lambda:edit_points(0)
        # 회전1
        self.tf1_btn2["command"] = None
        # 좌표수정2
        self.tf2_btn1["command"] = lambda:edit_points(1)
        # 회전2
        self.tf2_btn2["command"] = lambda:rotate_poly(0)
        # 좌표수정3
        self.tf3_btn1["command"] = None
        # 회전3
        self.tf3_btn2["command"] = lambda:rotate_poly(1)
        