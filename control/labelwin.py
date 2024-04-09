from gui.labelwin import configure

from utils import tool
from utils.text import *
from utils import process

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


class LabelWindow(tk.Toplevel):
    def __init__(self, origin_image, *args, points=None, logo_img_tk=None, callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("세부조정 창")
        self.focus() # 창 선택해줌
        self.grab_set() # 다른창 못건드림
        
        self.geometry(f"{self.winfo_screenwidth()//5*4}x{self.winfo_screenheight()//5*4}")
        self.resizable(False, False)
        self.overrideredirect(False)
        
        self.win_factor = self.winfo_screenheight() / 1980
        
        
        
        self.callback = callback
        self.logo_img_tk = logo_img_tk
        self.origin_image = origin_image
        self.points = points
        
        # GUI 및 bind
        configure(self)
        self.set_bind()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        
        Thread(target=self.canv.init_canv, args=(self.origin_image, self.points), daemon=True).start()
        
###########################################################################################
    def submit(self): # 여기선 박스 xyxy
        poly = self.canv.get_real_poly()
        self.callback(poly)
        self.destroy()
        
###########################################################################################
    def set_bind(self):
        def on_push(event):
            if not hasattr(event.widget, 'do_button'): return
            event.widget.configure(bg="#0153B0", fg="#FFFFFF")
            event.widget.do_button()

        def on_leave(event):
            if not hasattr(event.widget, 'do_button'): return
            event.widget.configure(bg="#393945", fg="#A6A6A6")
            
        # 클릭모션 부여
        btn_list = [self.br_btn1, self.br_btn2, self.br_btn3, 
                    self.br_btn4, self.br_btn5, self.br_btn6, self.apply_btn]
        for btn in btn_list:
            btn.bind("<Button-1>", on_push)
            btn.bind("<ButtonRelease-1>", on_leave)
        
        # 등록완료 버튼 기능
        self.apply_btn.do_button = self.submit
        # 축소 버튼 기능
        self.br_btn1.do_button = self.canv.zoom_out
        # 위 버튼 기능
        self.br_btn2.do_button = self.canv.move_up
        # 확대 버튼 기능
        self.br_btn3.do_button = self.canv.zoom_in
        # 왼쪽 버튼 기능
        self.br_btn4.do_button = self.canv.move_left
        # 아래 버튼 기능
        self.br_btn5.do_button = self.canv.move_down
        # 오른쪽 버튼 기능
        self.br_btn6.do_button = self.canv.move_right
