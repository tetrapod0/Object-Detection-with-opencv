from utils.poly import MultiPolyDetector
from utils.model import InferenceModel
from utils.logger import logger
from utils.text import *
from utils import tool
from utils import process

# from gui.labelwin import LabelWindow
from control.addwin import AddWindow
# from gui.login import LoginWindow
# from gui.pin import PinWindow
from gui.main import configure, colors

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.filedialog as filedialog
from tkinter import messagebox as mb

from collections import defaultdict
from threading import Thread, Lock
from PIL import ImageTk, Image
from copy import deepcopy
from queue import Queue
from glob import glob
import pandas as pd
import numpy as np
import traceback
import time
import json
import os

from json import JSONDecodeError

class MainWindow(tk.Tk):
    def __init__(self, *arg, nodb=False, hand=False, **kwargs):
        super().__init__(*arg, **kwargs)
        # self.iconbitmap(ICON_PATH)
        self.title(TITLE)
        
        # window size
        self.geometry(f"{self.winfo_screenwidth()//5*4}x{self.winfo_screenheight()//5*4}")
        self.resizable(False, False)
        self.overrideredirect(False)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # font size factor
        self.win_factor = self.winfo_screenheight() / 1980
        
        # for thread
        self.stop_signal = True
        self.raw_Q = Queue()
        self.analy_Q = Queue()
        self.draw_Q = Queue()
        self.image_Q = Queue()
        self.data_Q = Queue()
        self.thr_lock = Lock() # for serial
        
        
        self.treeview_cols = ['name']
        self.treeview_headnames = ['Name']
        self.table = pd.DataFrame([], columns=['name'])
        
        # gui
        configure(self)
        self.set_bind()
        # self.start_button.configure(text="..", command=lambda:time.sleep(0.1))
        self.change_frame(1)
        
        # 
        self.poly_detector = MultiPolyDetector(IMAGE_PATH, LABEL_PATH)
        self.ocr_engine = InferenceModel(OCR_MODEL_PATH)
        
        
        names = list(map(lambda x:x.split('.')[0], os.listdir(IMAGE_PATH)))
        for name in names:
            self.complete_apply(name, update_poly=False)
        self.poly_detector.update(pick_names=self.table['name'])
        
        # self.add_win = AddWindow('test', callback=self.complete_apply)
    
    #######################################################################
    def termination(self):
        self.stop_signal = True
        time.sleep(0.1)
        self.destroy()

    def on_closing(self):
        answer = mb.askquestion("종료하기", "종료 하시겠습니까?")
        if answer == "no": return
        
        Thread(target=self.termination, args=(), daemon=True).start()
    
    def trigger_btn(self):
        img = tool.get_url_img(IMG_URL)
        self.raw_Q.put(img)
    
    #######################################################################
    def stop(self):
        logger.info("Stop button clicked.")
        self.stop_signal = True
    
    #######################################################################
    def read_mode(self):
        # 판독영상으로 이동
        self.change_frame(0)
        
        # 시작
        self.stop_signal = False
        Thread(target=self.read_thread, args=(), daemon=True).start()

    def read_thread(self):
        tool.clear_Q(self.raw_Q)
        tool.clear_Q(self.analy_Q)
        tool.clear_Q(self.draw_Q)
        tool.clear_Q(self.image_Q)
        tool.clear_Q(self.data_Q)
        
        image_frame_list = [self.image_frame1, self.image_frame2, self.image_frame3, ]
        image_label_list = [self.image_label1, self.image_label2, self.image_label3, ]
        
        Thread(target=process.image_eater, args=(self, image_frame_list, image_label_list), daemon=True).start()
        Thread(target=process.data_eater, args=(self,), daemon=True).start()
        Thread(target=process.read, args=(self,), daemon=True).start()
        Thread(target=process.analysis, args=(self,), daemon=True).start()
        Thread(target=process.draw, args=(self,), daemon=True).start()

        self.start_button.configure(text="...", command=lambda:time.sleep(0.1))
        time.sleep(0.3)
        self.start_button.configure(text="■Stop", bg=colors[4], fg=colors[2], command=self.stop)
        
        # 중지 대기
        while not self.stop_signal: time.sleep(0.1)
        
        # GUI 초기화
        self.start_button.configure(text="...", command=lambda:time.sleep(0.1))
        time.sleep(0.3)
        self.start_button.configure(text="▶Start", bg=colors[1], fg=colors[2], command=self.read_mode)
        self.ok_label.configure(text='', bg=colors[2], fg=colors[1], font=self.font_func(250))
        
    #######################################################################
    def add_btn(self):
        name = self.entry.get()
        if not name.isalnum():
            mb.showinfo(title="", message="'Name' is alnum.")
            return
            
        if name in self.table.loc[:, 'name'].tolist():
            mb.showinfo(title="", message="Already Exist.")
            return
        
        self.add_win = AddWindow(name, callback=self.complete_apply)
        self.withdraw()
    
    def complete_apply(self, name, update_poly=True): # 등록하고 나올때
        self.deiconify()
        self.focus()
        self.grab_set()
        
        if name is None:return
    
        item_id = self.treeview.insert('', 'end')
        self.treeview.item(item_id, values=[name])
        self.table.loc[item_id, 'name'] = name
        
        # poly update
        if update_poly:
            self.poly_detector.update(pick_names=self.table['name'])
        
        logger.info('applied.')
    
    def delete_btn(self):
        selected_item = self.treeview.selection()
        if not selected_item:
            mb.showwarning(title="", message="Item is not selected.", parent=self)
            return
        item_id = selected_item[0]
        name = self.table.loc[item_id, 'name']
        
        img_path = os.path.join(IMAGE_PATH, name+'.png')
        json_path = os.path.join(LABEL_PATH, name+'.json')
        os.remove(img_path)
        os.remove(json_path)
        
        self.treeview.delete(item_id)
        self.table.drop(item_id, inplace=True)
        
        
    #######################################################################
    def change_frame(self, i):
        btn_list = [self.tf_btn1, self.tf_btn2, ]
        frame_list = [self.bottom_frame1, self.bottom_frame2, ]
        
        # 버튼 외관 변경
        for btn in btn_list: btn.configure(bg=colors[0], fg=colors[2])
        btn_list[i].configure(bg=colors[1], fg=colors[2])

        # 현재 프레임 변경
        for frame in frame_list: frame.place_forget()
        frame_list[i].place(relx=0.0, rely=0.4, relwidth=1, relheight=0.6)

    #######################################################################
    def set_bind(self):
        def select(event, i):
            if not self.stop_signal and (i==2 or i==3):
                mb.showinfo(title="", message="■중지 버튼을 먼저 눌러주세요.")
                return
            
            # 버튼과 프레임 변경
            self.change_frame(i)
        
        # 선택 부여
        self.tf_btn1.bind("<Button-1>", lambda _:select(_, 0)) # Image
        self.tf_btn2.bind("<Button-1>", lambda _:select(_, 1)) # Information
        
        
        