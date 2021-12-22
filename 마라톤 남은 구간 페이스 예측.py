#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math
import pandas as pd

marathon_2015_2017 = pd.read_csv("./data/marathon_2015_2017.csv")

record = pd.DataFrame(marathon_2015_2017,columns=['5K',  '10K',  '15K',  '20K', 'Half',  '25K',  '30K',  '35K',  '40K',  'Official Time']).sort_values(by=['Official Time'])

record_list = record.values.tolist()

xData = [5, 10, 15, 20, 21.098, 25, 30, 35, 40, 42.195 ]

fig = Figure(figsize=(6,6), dpi=100)
ax = fig.add_subplot(111)
t_xdata, t_ydata, ml_xdata, ml_ydata, p_xdata, p_ydata = [], [], [], [], [], []
ax.set_xlim(0, 45)
ax.set_ylim(0, 13000)
ax.set_xlabel('Distance(km)')
ax.set_ylabel('Time(Second)')
ax.set_title('Records of runner')
ln, = ax.plot([], [], linestyle=':')
dn, = ax.plot([], [], 'ro')
pn, = ax.plot([], [], 'bs')
t_a = 0

grad_fig = Figure(figsize=(6,6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(0, 5000)
grad_ax.set_ylim(0, 50000)
grad_ax.set_title('Cost Gradient Decent')
grad_ax.set_ylabel("Total Cost")
grad_ax.set_xlabel("Number of Traning")
g_xdata, g_ydata = [], []
gn, = grad_ax.plot([], [], 'ro')

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def init():
    t_a = int(t_aSpbox.get()) -1
    ax.set_title('Records of runner #'+str(t_a + 1))
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 13000)
    grad_ax.set_xlim(0, 5000)
    grad_ax.set_ylim(0, 50000)
    return dn,

def animateFrame(frame):
    t_a = int(t_aSpbox.get()) -1
    t_x = xData[int(frame)]
    t_y = record_list[t_a][int(frame)]
    t_xdata.append(t_x)
    t_ydata.append(t_y)  
    dn.set_data(t_xdata, t_ydata) 
    ax.annotate(seconds_to_hhmmss(t_y), (t_x, t_y), fontsize=8) 
    fig.canvas.draw()
    return dn,

def update(): 
    t_xdata.clear()
    t_ydata.clear()
    
    ani = FuncAnimation(fig, animateFrame, frames=np.linspace(0, len(xData)-1, len(xData)),
                        init_func=init, blit=True, repeat = False)
    fig.canvas.draw()

def learing():
    import tensorflow as tf
    
    t_a = int(t_aSpbox.get())
    t_t = int(t_tSpbox.get())
    t_r = float(t_rSpbox.get())
        
    x_train = [ i/10 for i in xData[0:7]]
    y_train = record_list[t_a-1][0:7]
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim =1))
    
    sgd = tf.keras.optimizers.SGD(lr = t_r)
    model.compile(loss = 'mse', optimizer = sgd)
    
    history = model.fit(x_train, y_train, epochs = t_t, batch_size=128)
    
    log_ScrolledText.insert(END, '\n\n')
    log_ScrolledText.insert(END, "%10s %4i %10s %6i %20s %10.8f" % ('\nRunner #', t_a, ', No. of train is', (t_t-1), ', learing rate is ', t_r)+'\n', 'TITLE')
    log_ScrolledText.insert(END, '\n\nCost Decent\n\n','HEADER')
    log_ScrolledText.insert(END, "%20s %20s" % ('Step', 'Cost')+'\n\n')
    for step in range(t_t):
        if step % 100 == 0:
            cost_val = history.history['loss'][step]
            g_xdata.append(step)
            g_ydata.append(cost_val)
            log_ScrolledText.insert(END, "%20i %20.5f" % (step, cost_val)+'\n')
    
    grad_ax.plot(g_xdata, g_ydata, 'ro')
    grad_ax.set_title('The minimum cost is %10.3f at %5i times' % (cost_val, step))
    grad_fig.canvas.draw()
    
    W_val = model.layers[0].get_weights()[0][0]
    b_val = model.layers[0].get_weights()[1]
    log_ScrolledText.insert(END, "%20s" % ('\n\nHypothesis = X * W + b\n\n'), 'HEADER')
    draw_hypothesis(W_val,b_val)
    
    log_ScrolledText.insert(END, "%20s" % ('\n\nRecords Prediction\n\n'), 'HEADER')
    log_ScrolledText.insert(END, "%20s %20s %20s %20s" % ('Distance(km)', 'Real record', 'ML Prediction', 'Variation(Second)')+'\n\n')
    for index in range(7, 10):
        x_value = xData[index] / 10
        p_xdata.append(xData[index])
        time = model.predict(np.array([x_value]))
        p_ydata.append(time[0][0])
        log_ScrolledText.insert(END,  "%20.3f %20s %20s %20i" % (xData[index], seconds_to_hhmmss(t_ydata[index]), seconds_to_hhmmss(time[0]), (t_ydata[index] - time[0]))+'\n')
    
    dn.set_data(t_xdata, t_ydata)
    pn.set_data(p_xdata, p_ydata)
    fig.canvas.draw()
    
main = Tk()
main.title("Marathon Records")
main.geometry()

label=Label(main, text='Marathon Records Prediction by Machine Learing')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=6)

t_aVal  = IntVar(value=1)
t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=0, to=len(record_list), increment=1, justify=RIGHT)
t_aSpbox.grid(row=1,column=1)
t_aLabel=Label(main, text='Rank of runner : ')                
t_aLabel.grid(row=1,column=0)

t_tVal  = IntVar(value=5000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
t_tSpbox.grid(row=1,column=3)
t_tLabel=Label(main, text='Number of train : ')                
t_tLabel.grid(row=1,column=2)

t_rVal  = DoubleVar(value=0.01)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=0.001, justify=RIGHT)
t_rSpbox.grid(row=1,column=5)
t_rLabel=Label(main, text='Learning rate : ')                
t_rLabel.grid(row=1,column=4)

Button(main,text="Get History", height=2,command=lambda:update()).grid(row=2, column=0, columnspan=3, sticky=(W, E))
Button(main,text="Machine Learing", height=2,command=lambda:learing()).grid(row=2, column=3, columnspan=3, sticky=(W, E))

canvas = FigureCanvasTkAgg(fig, main)
canvas.get_tk_widget().grid(row=3,column=0,columnspan=3) 

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=3,columnspan=3)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=6, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('RESULT', foreground='blue', font=("Helvetica", 12))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()


# In[ ]:





# In[ ]:




