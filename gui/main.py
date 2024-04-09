import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as font

# colors = ['#FFCCB2', '#FFB5A3', '#E3989A', '#B4828F', '#6E6875']
colors = ['#5F6C37', '#28361A', '#FEF9E2', '#DDA15F', '#BB6C25']

def configure(self):
    # bg
    bg_color = colors[2]
    self.configure(bg=bg_color)
    self.font_func = lambda x:font.Font(family='Helvetica', size=int(x*self.win_factor), weight='bold')
    
    
    # 스타일
    style = ttk.Style()
    style.theme_use("default")
    style.configure('Treeview', font=('Arial', int(22*self.win_factor), 'bold'), rowheight=38, 
                    background=colors[3], fieldbackground=colors[2], foreground=colors[2])
    style.configure('Treeview.Heading', font=('Arial', int(20*self.win_factor), 'bold'), 
                    background=colors[0], foreground=colors[2])
    style.map('Treeview', background=[('selected', colors[4])])
    style.layout('Vertical.TScrollbar', [
        ('Vertical.Scrollbar.trough', {'sticky': 'nswe', 'children': [
            ('Vertical.Scrollbar.uparrow', {'side': 'top', 'sticky': 'nswe'}),
            ('Vertical.Scrollbar.downarrow', {'side': 'bottom', 'sticky': 'nswe'}),
            ('Vertical.Scrollbar.thumb', {'sticky': 'nswe', 'unit': 1, 'children': [
                ('Vertical.Scrollbar.grip', {'sticky': ''})
                ]})
            ]})
        ])

    # title
    self.title_label = tk.Label(self, bd=0, relief="solid", font=self.font_func(50)) # "solid"
    self.title_label.place(relx=0.0, rely=0.0, relwidth=1, relheight=0.1)
    self.title_label.configure(text='Vision', bg=colors[3], fg=colors[2], anchor='center')
    self.back_btn = tk.Button(self, bd=3, text="Exit", command=self.on_closing, font=self.font_func(25))
    self.back_btn.place(relx=0.9, rely=0.0, relwidth=0.1, relheight=0.1)
    self.back_btn.configure(bg=colors[4], fg=colors[2], activebackground=colors[4], activeforeground=colors[2])

    # 상단프레임
    self.top_frame = tk.Frame(self, bd=1, relief="solid", bg=bg_color)
    self.top_frame.place(relx=0.0, rely=0.1, relwidth=1, relheight=0.3)

    # top_right
    self.frame = tk.Frame(self.top_frame, bd=1, relief="solid", bg=bg_color)
    self.frame.place(relx=0.75, rely=0.0, relwidth=0.25, relheight=1)

    # top_right - button
    self.tf_btn1 = tk.Label(self.frame, bd=1, relief="solid", anchor='center', text='Detection')
    self.tf_btn1.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=0.5)
    self.tf_btn2 = tk.Label(self.frame, bd=1, relief="solid", anchor='center', text='Information')
    self.tf_btn2.place(relx=0.0, rely=0.5, relwidth=1.0, relheight=0.5)
    for btn in [self.tf_btn1, self.tf_btn2, ]:
        btn.configure(bg=colors[0], fg=colors[2], font=self.font_func(40))
        # btn.configure(bg="#0153B0", fg="#FFFFFF")

    # top - OK
    self.ok_label = tk.Label(self.top_frame, relief="solid", bd=1, anchor='center') # "solid"
    self.ok_label.place(relx=0.5, rely=0.0, relwidth=0.25, relheight=1)
    self.ok_label.configure(text='OK', bg=colors[2], fg=colors[1], font=self.font_func(250))
    # self.ok_label.configure(text='NG', bg=colors[4], fg=colors[1], font=self.font_func(250))

    # top_left
    self.frame = tk.Frame(self.top_frame, bd=1, relief="solid", bg=bg_color)
    self.frame.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1)

    # top_left - button
    self.start_button = tk.Button(self.frame, bd=3)
    self.start_button.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)
    self.start_button['font'] = font.Font(family='Helvetica', size=int(70*self.win_factor), weight='bold')
    self.start_button.configure(text="▶Start", bg=colors[1], fg=colors[2], command=self.read_mode)
    # self.start_button.configure(text="■Stop", bg=colors[4], fg=colors[2], command=None)
    
    # top_left - label
    self.name_label1 = tk.Label(self.frame, anchor="center", text='Name', bg=colors[1], fg=colors[2])
    self.name_label2 = tk.Label(self.frame, anchor="center", text='Data', bg=colors[1], fg=colors[2])
    self.value_label1 = tk.Label(self.frame, anchor="center", text='-', bg=colors[0], fg=colors[2])
    self.value_label2 = tk.Label(self.frame, anchor="center", text='-', bg=colors[0], fg=colors[2])

    self.name_label1.place(relx=0.5, rely=0.125*0, relwidth=0.5, relheight=0.125)
    self.name_label2.place(relx=0.5, rely=0.125*2, relwidth=0.5, relheight=0.125)
    self.value_label1.place(relx=0.5, rely=0.125*1, relwidth=0.5, relheight=0.125)
    self.value_label2.place(relx=0.5, rely=0.125*3, relwidth=0.5, relheight=0.125)

    self.name_label2['font'] = self.font_func(20)
    self.name_label1['font'] = self.font_func(20)
    self.value_label1['font'] = self.font_func(20)
    self.value_label2['font'] = self.font_func(20)
    
    # top_left - trigger
    self.trigger_btn = tk.Button(self.frame, bd=1, text="Trigger", command=self.trigger_btn)
    self.trigger_btn.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.5)
    self.trigger_btn.configure(text='Shot', bg=colors[3], fg=colors[2], font=self.font_func(50))
    # self.trigger_btn.place_forget()



    # bottom - init
    self.bottom_frame0 = tk.Frame(self, bd=1, relief="solid", bg=bg_color)
    self.bottom_frame0.place(relx=0.0, rely=0.4, relwidth=1, relheight=0.6)
    self.hi_label = tk.Label(self.bottom_frame0, relief="solid", bd=1, anchor='center') # "solid"
    self.hi_label.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    self.hi_label['font'] = font.Font(family='Calibri', size=int(50*self.win_factor), weight='bold')
    self.hi_label.configure(text='Hello.', bg=bg_color, fg=colors[1])
    # self.bottom_frame1.place_forget()
    # self.bottom_frame1.place(relx=0.0, rely=0.4, relwidth=1, relheight=0.6)

    # bottom - Image
    self.bottom_frame1 = tk.Frame(self, bd=1, relief="solid", bg=bg_color)
    self.bottom_frame1.place(relx=0.0, rely=0.4, relwidth=1, relheight=0.6)
    self.bottom_frame1.place_forget()

    # bottom - Image - frame1
    self.image_frame1 = tk.Frame(self.bottom_frame1, bd=1, relief="solid") # "solid"
    self.image_frame1.place(relx=0.0, rely=0.0, relwidth=0.55, relheight=1)
    self.image_label1_ = tk.Label(self.image_frame1, anchor="center", text='No Image')
    self.image_label1_.configure(fg=colors[1], bg=bg_color)
    self.image_label1_.place(relx=0, rely=0, relwidth=1, relheight=1)
    self.image_label1 = tk.Label(self.image_frame1)
    self.image_label1.configure(fg=colors[1], bg=bg_color)
    self.image_label1.pack(expand=True, fill="both")
    self.image_label1.pack_forget()

    # bottom - Image - frame2
    self.temp = tk.Label(self.bottom_frame1, anchor="center", text='Image')
    self.temp.place(relx=0.55, rely=0.0, relwidth=0.45, relheight=0.1)
    self.temp.configure(bg=colors[4], fg=colors[2], font=self.font_func(25))
    self.image_frame2 = tk.Frame(self.bottom_frame1, bd=1, relief="solid") # "solid"
    self.image_frame2.place(relx=0.55, rely=0.1, relwidth=0.45, relheight=0.6)
    self.image_label2_ = tk.Label(self.image_frame2, anchor="center", text='No Image')
    self.image_label2_.configure(fg=colors[1], bg=bg_color)
    self.image_label2_.place(relx=0, rely=0, relwidth=1, relheight=1)
    self.image_label2 = tk.Label(self.image_frame2)
    self.image_label2.configure(fg=colors[1], bg=bg_color)
    self.image_label2.pack(expand=True, fill="both")
    self.image_label2.pack_forget()

    # bottom - Image - frame3
    self.temp = tk.Label(self.bottom_frame1, anchor="center", text='Data')
    self.temp.place(relx=0.55, rely=0.7, relwidth=0.45, relheight=0.1)
    self.temp.configure(bg=colors[4], fg=colors[2], font=self.font_func(25))
    self.image_frame3 = tk.Frame(self.bottom_frame1, bd=1, relief="solid") # "solid"
    self.image_frame3.place(relx=0.55, rely=0.8, relwidth=0.45, relheight=0.2)
    self.image_label3_ = tk.Label(self.image_frame3, anchor="center", text='No Image')
    self.image_label3_.configure(fg=colors[1], bg=bg_color)
    self.image_label3_.place(relx=0, rely=0, relwidth=1, relheight=1)
    self.image_label3 = tk.Label(self.image_frame3)
    self.image_label3.configure(fg=colors[1], bg=bg_color)
    self.image_label3.pack(expand=True, fill="both")
    self.image_label3.pack_forget()


    # bottom - Information
    self.bottom_frame2 = tk.Frame(self, bd=10, relief=None, bg=bg_color)
    self.bottom_frame2.place(relx=0.0, rely=0.4, relwidth=1, relheight=0.6)
    self.bottom_frame2.place_forget()
    
    # bottom - Information - button
    self.frame = tk.Frame(self.bottom_frame2, bg=bg_color, bd=1, relief="solid")
    self.frame.place(relx=0.49, rely=0.0, relwidth=0.51, relheight=0.1)
    self.entry = tk.Entry(self.frame, font=self.font_func(30))
    self.entry.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1)
    self.btn = tk.Button(self.frame, text='Add', font=self.font_func(30), command=self.add_btn)
    self.btn.place(relx=0.5, rely=0.0, relwidth=0.25, relheight=1)
    self.btn.configure(bg=colors[1], fg=colors[2], activebackground=colors[2], activeforeground=colors[1])
    self.btn = tk.Button(self.frame, text='Delete', font=self.font_func(30), command=self.delete_btn)
    self.btn.place(relx=0.75, rely=0.0, relwidth=0.25, relheight=1)
    self.btn.configure(bg=colors[1], fg=colors[2], activebackground=colors[2], activeforeground=colors[1])
    # self.btn = tk.Button(self.frame, text='Apply', font=self.font_func(30), command=self.apply_btn)
    # self.btn.place(relx=0.8, rely=0.0, relwidth=0.2, relheight=1)
    # self.btn.configure(bg=colors[1], fg=colors[2], activebackground=colors[2], activeforeground=colors[1])
    
    # bottom - Information - treeview
    self.treeview = ttk.Treeview(self.bottom_frame2)
    self.treeview['columns'] = self.treeview_cols
    self.treeview['show'] = 'headings'
    self.treeview.place(relx=0.0, rely=0.1, relwidth=0.98, relheight=0.9)
    for col, name in zip(self.treeview['columns'], self.treeview_headnames):
        self.treeview.heading(col, text=name)
    
    # bottom - Information - treeview - scroll
    self.scrollbar = tk.Scrollbar(self.bottom_frame2, orient="vertical", command=self.treeview.yview)
    self.scrollbar.place(relx=0.98, rely=0.1, relwidth=0.02, relheight=0.9)
    self.treeview.configure(yscrollcommand=self.scrollbar.set)
    
    
    # item_id = self.treeview.insert('', 'end')
    # self.treeview.item(item_id, values=['test', 'test'])
