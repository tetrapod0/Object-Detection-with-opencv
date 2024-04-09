import tkinter as tk
import tkinter.font as font

colors = ['#5F6C37', '#28361A', '#FEF9E2', '#DDA15F', '#BB6C25']

###########################################################################################
def configure(self):
    # bg
    bg_color = colors[2]
    self.configure(bg=bg_color)
    self.font_func = lambda x:font.Font(family='Helvetica', size=int(x*self.win_factor), weight='bold')

    # title
    self.title_label = tk.Label(self, bd=0, relief="solid", font=self.font_func(50)) # "solid"
    self.title_label.place(relx=0.0, rely=0.0, relwidth=1, relheight=0.1)
    self.title_label.configure(text='Apply Window', bg=colors[3], fg=colors[2], anchor='center')
    self.back_btn = tk.Button(self, bd=3, text="Back", command=self.on_closing, font=self.font_func(25))
    self.back_btn.place(relx=0.9, rely=0.0, relwidth=0.1, relheight=0.1)
    self.back_btn.configure(bg=colors[4], fg=colors[2], activebackground=colors[2], activeforeground=colors[4])
    
#############################

    # 상단 프레임1
    self.top1_frame = tk.Frame(self, bd=10, relief=None, bg=bg_color)
    self.top1_frame.place(relx=0.0, rely=0.1, relwidth=0.33, relheight=0.6)

    # 상단 프레임1 - 제목라벨
    self.tf1_label = tk.Label(self.top1_frame, bd=0, relief="solid") # "solid"
    self.tf1_label.place(relx=0.0, rely=0.0, relwidth=1, relheight=0.1)
    self.tf1_label.configure(bg=colors[0], fg=colors[2], text='Raw Image', font=self.font_func(25))

    # 상단 프레임1 - 이미지프레임
    self.tf1_frame = tk.Frame(self.top1_frame, bd=1, relief="solid", bg=bg_color)
    self.tf1_frame.place(relx=0.0, rely=0.1, relwidth=1, relheight=0.7)

    # 상단 프레임1 - 이미지프레임 - 이미지라벨
    self.tf1f_label_ = tk.Label(self.tf1_frame, anchor="center", text='No Image')
    self.tf1f_label_.configure(fg=colors[1], bg=bg_color)
    self.tf1f_label_.place(relx=0, rely=0, relwidth=1, relheight=1)
    self.tf1f_label = tk.Label(self.tf1_frame)
    self.tf1f_label.configure(bg=bg_color)
    self.tf1f_label.pack(expand=True, fill="both")
    self.tf1f_label.pack_forget()

    # 상단 프레임1 - 수정버튼
    self.tf1_btn1 = tk.Button(self.top1_frame, bd=1, text="Edit", command=None, font=self.font_func(35))
    self.tf1_btn1.place(relx=0.0, rely=0.8, relwidth=0.5, relheight=0.2)
    self.tf1_btn1.configure(bg=colors[1], fg=colors[2], activebackground=colors[2], activeforeground=colors[1])
    self.tf1_btn2 = tk.Button(self.top1_frame, bd=1, text="", command=None, font=self.font_func(35))
    self.tf1_btn2.place(relx=0.5, rely=0.8, relwidth=0.5, relheight=0.2)
    self.tf1_btn2.configure(bg=colors[1], fg=colors[2], activebackground=colors[2], activeforeground=colors[1])


    # 상단 프레임2
    self.top2_frame = tk.Frame(self, bd=10, relief=None, bg=bg_color)
    self.top2_frame.place(relx=0.33, rely=0.1, relwidth=0.33, relheight=0.6)

    # 상단 프레임2 - 제목라벨
    self.tf2_label = tk.Label(self.top2_frame, bd=0, relief="solid") # "solid"
    self.tf2_label.place(relx=0.0, rely=0.0, relwidth=1, relheight=0.1)
    self.tf2_label.configure(bg=colors[0], fg=colors[2], text='Object Image', font=self.font_func(25))

    # 상단 프레임2 - 이미지프레임
    self.tf2_frame = tk.Frame(self.top2_frame, bd=1, relief="solid", bg=bg_color)
    self.tf2_frame.place(relx=0.0, rely=0.1, relwidth=1, relheight=0.7)

    # 상단 프레임2 - 이미지프레임 - 이미지라벨
    self.tf2f_label_ = tk.Label(self.tf2_frame, anchor="center", text='No Image')
    self.tf2f_label_.configure(fg=colors[1], bg=bg_color)
    self.tf2f_label_.place(relx=0, rely=0, relwidth=1, relheight=1)
    self.tf2f_label = tk.Label(self.tf2_frame)
    self.tf2f_label.configure(bg=bg_color)
    self.tf2f_label.pack(expand=True, fill="both")
    self.tf2f_label.pack_forget()

    # 상단 프레임2 - 수정버튼
    self.tf2_btn1 = tk.Button(self.top2_frame, bd=1, text="Edit", command=None, font=self.font_func(35))
    self.tf2_btn1.place(relx=0.0, rely=0.8, relwidth=0.5, relheight=0.2)
    self.tf2_btn1.configure(bg=colors[1], fg=colors[2], activebackground=colors[2], activeforeground=colors[1])
    self.tf2_btn2 = tk.Button(self.top2_frame, bd=1, text="Rotate", command=None, font=self.font_func(35))
    self.tf2_btn2.place(relx=0.5, rely=0.8, relwidth=0.5, relheight=0.2)
    self.tf2_btn2.configure(bg=colors[1], fg=colors[2], activebackground=colors[2], activeforeground=colors[1])


    # 상단 프레임3
    self.top3_frame = tk.Frame(self, bd=10, relief=None, bg=bg_color)
    self.top3_frame.place(relx=0.66, rely=0.1, relwidth=0.34, relheight=0.6)

    # 상단 프레임3 - 제목라벨
    self.tf3_label = tk.Label(self.top3_frame, bd=0, relief="solid") # "solid"
    self.tf3_label.place(relx=0.0, rely=0.0, relwidth=1, relheight=0.1)
    self.tf3_label.configure(bg=colors[0], fg=colors[2], text='Data Image', font=self.font_func(25))

    # 상단 프레임3 - 이미지프레임
    self.tf3_frame = tk.Frame(self.top3_frame, bd=1, relief="solid", bg=bg_color)
    self.tf3_frame.place(relx=0.0, rely=0.1, relwidth=1, relheight=0.7)

    # 상단 프레임3 - 이미지프레임 - 이미지라벨
    self.tf3f_label_ = tk.Label(self.tf3_frame, anchor="center", text='No Image')
    self.tf3f_label_.configure(fg=colors[1], bg=bg_color)
    self.tf3f_label_.place(relx=0, rely=0, relwidth=1, relheight=1)
    self.tf3f_label = tk.Label(self.tf3_frame)
    self.tf3f_label.configure(bg=bg_color)
    self.tf3f_label.pack(expand=True, fill="both")
    self.tf3f_label.pack_forget()

    # 상단 프레임3 - 수정버튼
    self.tf3_btn1 = tk.Button(self.top3_frame, bd=1, text="", command=None, font=self.font_func(35))
    self.tf3_btn1.place(relx=0.0, rely=0.8, relwidth=0.5, relheight=0.2)
    self.tf3_btn1.configure(bg=colors[1], fg=colors[2], activebackground=colors[2], activeforeground=colors[1])
    self.tf3_btn2 = tk.Button(self.top3_frame, bd=1, text="Rotate", command=None, font=self.font_func(35))
    self.tf3_btn2.place(relx=0.5, rely=0.8, relwidth=0.5, relheight=0.2)
    self.tf3_btn2.configure(bg=colors[1], fg=colors[2], activebackground=colors[2], activeforeground=colors[1])


    # 중단 프레임
    self.mid_frame = tk.Frame(self, bd=10, relief=None, bg=bg_color)
    self.mid_frame.place(relx=0.0, rely=0.7, relwidth=1, relheight=0.1)

    # 중단 프레임 - 라벨
    self.mf_label_ = tk.Label(self.mid_frame, anchor="center", text='Name')
    self.mf_label_.place(relx=0, rely=0, relwidth=0.2, relheight=1)
    self.mf_label_.configure(bg=colors[4], fg=colors[2], font=self.font_func(40))
    self.mf_label = tk.Label(self.mid_frame, anchor="center", text=self.name)
    self.mf_label.place(relx=0.2, rely=0, relwidth=0.8, relheight=1)
    self.mf_label.configure(bg=colors[3], fg=colors[2], font=self.font_func(40))


    # 하단 프레임
    self.bottom_frame = tk.Frame(self, bd=10, relief=None, bg=bg_color)
    self.bottom_frame.place(relx=0.0, rely=0.8, relwidth=1, relheight=0.2)

    # 하단 프레임 - 버튼들
    self.bf_btn1 = tk.Label(self.bottom_frame, bd=1, relief="solid", anchor='center', text='Data\nSwitch')
    self.bf_btn1.place(relx=0.2*0, rely=0.0, relwidth=0.2, relheight=1)
    self.bf_btn3 = tk.Label(self.bottom_frame, bd=1, relief="solid", anchor='center', text='Shot\nImage')
    self.bf_btn3.place(relx=0.2*1, rely=0.0, relwidth=0.2, relheight=1)
    self.bf_btn4 = tk.Label(self.bottom_frame, bd=1, relief="solid", anchor='center', text='Open\nFile')
    self.bf_btn4.place(relx=0.2*2, rely=0.0, relwidth=0.2, relheight=1)
    self.bf_btn5 = tk.Label(self.bottom_frame, bd=1, relief="solid", anchor='center', text='Init')
    self.bf_btn5.place(relx=0.2*3, rely=0.0, relwidth=0.2, relheight=1)
    self.bf_btn6 = tk.Label(self.bottom_frame, bd=1, relief="solid", anchor='center', text='Complete')
    self.bf_btn6.place(relx=0.2*4, rely=0.0, relwidth=0.2, relheight=1)
    for btn in [self.bf_btn1, self.bf_btn3, self.bf_btn4, 
                self.bf_btn5, self.bf_btn6]:
        btn.configure(bg=colors[1], fg=colors[2], font=self.font_func(40))
