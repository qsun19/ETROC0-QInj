import os, sys
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.font_manager as font_manager
from tqdm.autonotebook import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import time
from scipy.optimize import curve_fit
from scipy.stats import norm
import scipy.stats

tdc_bin = 5e-3  ## in ns

def normalizeY(points):
    scale = points[np.argmax(np.abs(points))]
    return np.multiply(points, 1.0/scale)

def crossFinder(y): 
        jmin = y.argmin()
        jmax = y.argmax()
        x = np.where(np.diff(np.sign(y)))[0]
        arr1 = np.where(x<jmax)
        if len(arr1[0]) == 0: return -1
        arr1 = arr1[0][-1]
        return int(x[arr1])

def simulateCFT(xs, ys, dt):
        frac = 0.5
        delayInterval = int(1000/dt) # delay 1000 ps
        f = InterpolatedUnivariateSpline(xs, ys)
        def simCFT( x):
            return f(x-delayInterval*dt)-frac*f(x)
        return simCFT

def getCFTiming(t, y):
        y = normalizeY(y)
        dt = t[1]-t[0]
        print('dt is', dt)
        ntrun =int(np.floor(2000/dt))
        print('ntrun is', ntrun)
        f = simulateCFT(t, y, dt)
        yy = f(t[ntrun:-ntrun])
        x0 = crossFinder(yy) 
        if x0 < 0 : return -1
        inte = [t[x0+ntrun], t[x0+1+ntrun]]
        x1 = inte[0]
        x2 = inte[1]
        k = (f(x2)-f(x1))/(x2-x1)
        cft = x1-f(x1)/k
        return cft

def extract_dataset(filename, dataset_name='waveform'):
    f = h5py.File(filename,'r')
    dataset = f[dataset_name]
    attrs_out = dict(dataset.attrs)
    ymults = [dataset.attrs['vertical{0}'.format(i+1)][0] for i in range(4)]
    yzeros = [dataset.attrs['vertical{0}'.format(i+1)][1] for i in range(4)]
    npoints = dataset.attrs['nPt']
    events = dataset.shape[1]//npoints
    chmask = dataset.attrs['chmask']  
    data_out = np.zeros(4*dataset.shape[1]).reshape((4, events, npoints))
    ich = 0
    for i in range(4):
        if chmask[i]:
            data_out[i] = (dataset[ich].reshape(events, npoints) - yzeros[i])*ymults[i]
            ich += 1
    f.close()
    return data_out, attrs_out
    
def extract_dataset_3ch(filename, dataset_name='waveform'):
    f = h5py.File(filename,'r')
    dataset = f[dataset_name]
    attrs_out = dict(dataset.attrs)
    ymults = [dataset.attrs['vertical{0}'.format(i+1)][0] for i in range(3)]
    yzeros = [dataset.attrs['vertical{0}'.format(i+1)][1] for i in range(3)]
    npoints = dataset.attrs['nPt']
    events = dataset.shape[1]//npoints
    chmask = dataset.attrs['chmask']  
    data_out = np.zeros(3*dataset.shape[1]).reshape((3, events, npoints))
    ich = 0
    for i in range(3):
        if chmask[i]:
            data_out[i] = (dataset[ich].reshape(events, npoints) - yzeros[i])*ymults[i]
            ich += 1
    f.close()
    return data_out, attrs_out

def calculate_voltages(v_in, pedestal_length=400):
    v_pedestal = [np.mean(v_in[i][0:pedestal_length]) for i in range(len(v_in))]
    v_pedestal_m = np.mean(v_pedestal)
    v_shift = [0]*len(v_in)
    v_amp = [0]*len(v_in)
    for event in range(0,len(v_in)):
        v_shift[event] = v_in[event] - v_pedestal_m
		#v_shift[event] = v_in[event] - v_pedestal[event]
        v_amp[event] = v_pedestal_m - np.min(v_in[event])

    return v_shift, v_amp

def calculate_charge(v_in, transCond, time):
    y1_integral = [0]*len(v_in)
    item_num = 0
    #y1_integral = integrate.simps(v_in/transCond*1e15, time*1e-9, axis=-1)/-10
    y1_integral = integrate.simps(v_in/transCond*1e15, time*1e-9)/-9.6
    return y1_integral

def plot_waveform(time, voltage, xlable="Time(ns)", ylable="Voltage(V)", title="Raw Data ch1", pdf=False, pic=False):
    fig, ax1 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(voltage))):
        ax1.plot(time, voltage[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax1.grid()
    ax1.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch1')
    if pdf==True:
        pp.savefig(fig)
        pp.close()
    if pic==True:
        plt.show()
    plt.close(fig)

def plot_waveforms(time, v_ch1, v_ch2, v_ch3, v_ch4, xlable="Time(ns)", ylable="Voltage(V)", 
                   title="Raw Data ch1", pdf=False, pic=False):
    fig, ax1 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(v_ch1))):
        ax1.plot(time, v_ch1[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax1.grid()
    ax1.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch1')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax2 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(v_ch2))):
        ax2.plot(time, v_ch2[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax2.grid()
    ax2.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch2')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax3 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(v_ch3))):
        ax3.plot(time, v_ch3[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax3.grid()
    ax3.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch3')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax4 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(v_ch4))):
        ax4.plot(time, v_ch4[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax4.grid()
    ax4.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch4')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
#     pp.close()

def plot_amplitudes(ampl_ch1, ampl_ch2, ampl_ch3, 
                   pic=True, pdf=False, num_bins=200, range_ampl=(0.0, 0.4)):
    fig, ax1 = plt.subplots(dpi=200)
    ax1.hist(ampl_ch1, num_bins, range=range_ampl, density=False,label='#event = %d\n#bin = %d'%(len(ampl_ch1),num_bins))
    ax1.legend()
    ax1.grid()
    ax1.set(xlabel='Amp(V)', ylabel='Occurance',
       title='Ampiltude Distribution of Ch1')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax2 = plt.subplots(dpi=200)
    ax2.hist(ampl_ch2, num_bins, range=range_ampl, density=False,label='#event = %d\n#bin = %d'%(len(ampl_ch2),num_bins))
    ax2.legend()
    ax2.grid()
    ax2.set(xlabel='Amp(V)', ylabel='Occurance',
       title='Ampiltude Distribution of Ch2')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax3 = plt.subplots(dpi=200)
    ax3.hist(ampl_ch3, num_bins, range=range_ampl, density=False,label='#event = %d\n#bin = %d'%(len(ampl_ch3),num_bins))
    ax3.legend()
    ax3.grid()
    ax3.set(xlabel='Amp(V)', ylabel='Occurance',
       title='Ampiltude Distribution of Ch3')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    

def plot_charges(q_ch1, q_ch2, q_ch3, 
                   pic=True, pdf=False, num_bins=200, range_q=(-5, 30)):
    fig, ax1 = plt.subplots(dpi=200)
    ax1.hist(q_ch1, num_bins, range=range_q, density=False,label='#event = %d\n#bin = %d'%(len(q_ch1),num_bins))
    ax1.legend()
    ax1.grid()
    ax1.set(xlabel='Charge(fC)', ylabel='Occurance',
       title='Charge Distribution of Ch1')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax2 = plt.subplots(dpi=200)
    ax2.hist(q_ch2, num_bins, range=range_q, density=False,label='#event = %d\n#bin = %d'%(len(q_ch2),num_bins))
    ax2.legend()
    ax2.grid()
    ax2.set(xlabel='Charge(fC)', ylabel='Occurance',
       title='Charge Distribution of Ch2')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax3 = plt.subplots(dpi=200)
    ax3.hist(q_ch3, num_bins, range=range_q, density=False,label='#event = %d\n#bin = %d'%(len(q_ch3),num_bins))
    ax3.legend()
    ax3.grid()
    ax3.set(xlabel='Charge(fC)', ylabel='Occurance',
       title='Charge Distribution of Ch3')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
def calculate_time(v_in, time, th_cfd=0.5, tdc_bin=0.005, tdc_start = 40):
    eventLen = len(v_in)
    interp_signal = np.repeat(None, eventLen)
    item_num = 0
    for iset in range(eventLen):
        interp_signal[item_num] = InterpolatedUnivariateSpline(time, v_in[item_num])
        item_num = item_num + 1
    
    v_pk_range = np.linspace(40, 80, 10000)
    v_pk = [np.min(interp_signal[event](v_pk_range)) for event in range (0,eventLen)]
    
    v_th_cfd = [0]*eventLen
    for elem in range(0,eventLen):
        v_th_cfd[elem] = 0 - th_cfd*(0 - v_pk[elem])
    
    t_cfd = [0] * eventLen
    for event in tqdm(range (0,eventLen)):
        t_cfd[event] = tdc_start
        while(interp_signal[event](t_cfd[event]) > v_th_cfd[event]):
            t_cfd[event] = t_cfd[event] + tdc_bin
    
    print('t searching is done')
    std_t_cfd = np.std(t_cfd)
    std_t_cfd = std_t_cfd * 1e3
    mean_t_cfd = np.mean(t_cfd)
    return t_cfd, std_t_cfd, mean_t_cfd
    
def calculate_time_qinj_amp(v_in, time, vth=0.5, tdc_bin=0.005, tdc_start = 40):
    eventLen = len(v_in)
    interp_signal = np.repeat(None, eventLen)
    item_num = 0
    for iset in range(eventLen):
        interp_signal[item_num] = InterpolatedUnivariateSpline(time, v_in[item_num])
        item_num = item_num + 1
           
    t_cross = [0] * eventLen
    for event in tqdm(range (0,eventLen)):
        t_cross[event] = tdc_start
        while(interp_signal[event](t_cross[event]) > vth):
            t_cross[event] = t_cross[event] + tdc_bin
    
    print('t searching is done')
    std_t_cross = np.std(t_cross)
    std_t_cross = std_t_cross * 1e3       # in pico second
    mean_t_cross = np.mean(t_cross)
    return t_cross, std_t_cross, mean_t_cross

def calculate_time_qinj_trigger(v_in, time, vth=0.5, tdc_bin=0.005, tdc_start = 20):
    eventLen = len(v_in)
    interp_signal = np.repeat(None, eventLen)
    item_num = 0
    for iset in range(eventLen):
        interp_signal[item_num] = InterpolatedUnivariateSpline(time, v_in[item_num])
        item_num = item_num + 1
           
    t_leading = [0] * eventLen
    
    for event in tqdm(range (0,eventLen)):
        t_leading[event] = tdc_start
        while(interp_signal[event](t_leading[event]) < vth):
            t_leading[event] = t_leading[event] + tdc_bin
    t_trailing = [0] * eventLen
    
    print('trigger crossing searching is done')
    return t_leading
    
def calculate_time_qinj_discri(v_in, time, vth=0.5, tdc_bin=0.005, tdc_leading_start = 20, tdc_trailing_start = 30):
    eventLen = len(v_in)
    interp_signal = np.repeat(None, eventLen)
    item_num = 0
    for iset in range(eventLen):
        interp_signal[item_num] = InterpolatedUnivariateSpline(time, v_in[item_num])
        item_num = item_num + 1
           
    t_leading = [0] * eventLen
    
    for event in tqdm(range (0,eventLen)):
        t_leading[event] = tdc_leading_start
        while(interp_signal[event](t_leading[event]) < vth):
            t_leading[event] = t_leading[event] + tdc_bin
    t_trailing = [0] * eventLen
    for event in tqdm(range (0,eventLen)):
        t_trailing[event] = tdc_trailing_start
        while(interp_signal[event](t_trailing[event]) < vth):
            t_trailing[event] = t_trailing[event] - tdc_bin
    
    print('discriminator crossing searching is done')
    return t_leading, t_trailing

def calculate_time_cfd1(v_in, time, cfd_dly=0.1, cfd_gain = 0.5, tdc_bin=0.005, tdc_start = 40, 
                        title_plot='ch1 cfd', pic=False, pdf = True):
    eventLen = len(v_in)
    interp_signal = np.repeat(None, eventLen)
    interp_signal_dly = np.repeat(None, eventLen)
    interp_signal_scaled = np.repeat(None, eventLen)
    item_num = 0
    for iset in range(eventLen):
        interp_signal[item_num] = InterpolatedUnivariateSpline(time, v_in[item_num])
        interp_signal_dly[item_num] = InterpolatedUnivariateSpline(time + cfd_dly, v_in[item_num])
        interp_signal_scaled[item_num] = InterpolatedUnivariateSpline(time, (-cfd_gain) *v_in[item_num]) 
        item_num = item_num + 1
    
    t_cfd = [0] * eventLen
    for event in tqdm(range (0,eventLen)):
        t_cfd[event] = tdc_start
        while((interp_signal_scaled[event](t_cfd[event]) + interp_signal_dly[event](t_cfd[event])) < 0):
            t_cfd[event] = t_cfd[event] - tdc_bin
    
#     tmin = np.min(time)
#     tmax = np.max(time)
#     xs = np.linspace(tmin, tmax, 10000)
    
#     fig, ax1 = plt.subplots(dpi=200)
#     for ab in range(0,eventLen):
#         ax1.plot(xs, interp_signal_scaled[ab](xs) + interp_signal_dly[ab](xs))
#     ax1.set_xlim(left=48,right=60)
#     ax1.set_ylim(bottom=-0.20,top=0.20)
#     ax1.grid()
#     ax1.set(xlabel='Time(ns)', ylabel='Voltage(V)',
#            title=title_plot)
#     if pdf:
#         pp.savefig(fig)
#         pp.close()
#     if pic:
#         plt.show()
#     plt.close(fig)
    
    std_t_cfd = np.std(t_cfd)
    std_t_cfd = std_t_cfd * 1e3
    mean_t_cfd = np.mean(t_cfd)
    return t_cfd, std_t_cfd, mean_t_cfd

def calculate_time_cfd2(v_in, time, cfd_dly=0.1, cfd_gain = 0.5, tdc_bin=0.005, tdc_start = 40, 
                        title_plot='ch1 cfd', pic=False, pdf = True):
    eventLen = len(v_in)
    t_cfd = []*eventLen
    for inum in range(eventLen):
        t_cfd[inum] = getCFTiming(time*1000, v_in[inum])
    
    std_t_cfd = np.std(t_cfd)
    std_t_cfd = std_t_cfd * 1e3
    mean_t_cfd = np.mean(t_cfd)
    return t_cfd, std_t_cfd, mean_t_cfd
    

def plot_times(t_ch1, t_ch2, t_ch3, std_t1, std_t2, std_t3, mean_t1, mean_t2, mean_t3,
                   pic=True, pdf=False, num_bins=200, range_t=None):
    fig, ax1 = plt.subplots(dpi=200)
    ax1.hist(t_ch1, num_bins, range=range_t, density=False,label = 
             '#event = %d\n#bin = %d\nstd = %.2f ps\nmean = %.2f ns'%(len(t_ch1),num_bins,std_t1, mean_t1))
    ax1.legend()
    ax1.grid()
    ax1.set(xlabel='TOA(ns)', ylabel='Occurance',
       title='TOA of Ch1')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax2 = plt.subplots(dpi=200)
    ax2.hist(t_ch2, num_bins, range=range_t, density=False,label = 
             '#event = %d\n#bin = %d\nstd = %.2f ps\nmean = %.2f ns'%(len(t_ch2),num_bins,std_t2, mean_t2))
    ax2.legend()
    ax2.grid()
    ax2.set(xlabel='TOA(ns)', ylabel='Occurance',
       title='TOA of Ch2')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax3 = plt.subplots(dpi=200)
    ax3.hist(t_ch3, num_bins, range=range_t, density=False,label = 
             '#event = %d\n#bin = %d\nstd = %.2f ps\nmean = %.2f ns'%(len(t_ch3),num_bins,std_t3, mean_t3))
    ax3.legend()
    ax3.grid()
    ax3.set(xlabel='TOA(ns)', ylabel='Occurance',
       title='TOA of Ch3')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)


def analyze_singlePointHV(file_in, path, file_index, file_num, tdc_bin=5e-3, cfd_ratio=0.5, transCond=4.4e3, trigger_ch1 = -0.03, trigger_ch2 = -0.03,
                         trigger_ch3 = -0.03, trigger_sat = -0.25, pedestal_length = 400, signal_end = 800):
    file = path + file_in
    data, attrs = extract_dataset(file)
    events_t = attrs['readout_size']
    npoints = attrs['nPt']
    x = np.arange(npoints)
    dt = npoints = attrs['dt']
    
    filelist = []
    filelist_to_use = [file_in]
    for apath in filelist_to_use:
        if apath.find(".hdf5") > -1:
            filelist.append(os.path.join(path,apath))
    y_ch1 = []
    y_ch2 = []
    y_ch3 = []
    y_ch4 = []
    for item in tqdm(filelist):
        data, attrs = extract_dataset(item)
        for event in range(len(data[0])):
            y_ch1.append(data[0][event])
        for event in range(len(data[1])):
            y_ch2.append(data[1][event])
        for event in range(len(data[2])):
            y_ch3.append(data[2][event])
        for event in range(len(data[3])):
            y_ch4.append(data[3][event])
    
    events_ch1 = len(y_ch1)
    events_ch2 = len(y_ch2)
    events_ch3 = len(y_ch3)
    events_ch4 = len(y_ch4)

    events_t = events_ch1
    time = x*dt*1e9
    
#     print('plot raw waveforms',file_index,'of',file_num)
#     plot_waveforms(time,y_ch1, y_ch2, y_ch3, y_ch4, pic=False, pdf=False)
    
    ####### apply software trigger  ############

    v_pk_raw_ch1 = [np.min(y_ch1[event][pedestal_length:signal_end]) for event in range (0,events_t)]
    v_pk_raw_ch2 = [np.min(y_ch2[event][pedestal_length:signal_end]) for event in range (0,events_t)]
    v_pk_raw_ch3 = [np.min(y_ch3[event][pedestal_length:signal_end]) for event in range (0,events_t)]


    event_soft_triger = 0
    for event in range(0,events_t):
        if (v_pk_raw_ch1[event] < trigger_ch1 and \
            v_pk_raw_ch2[event] < trigger_ch2 and \
            v_pk_raw_ch3[event] < trigger_ch3 and \
            v_pk_raw_ch1[event] > trigger_sat and \
            v_pk_raw_ch2[event] > trigger_sat and \
            v_pk_raw_ch3[event] > trigger_sat  ):
            event_soft_triger = event_soft_triger + 1

    y_soft_trigger_ch4 = [0]*(event_soft_triger)
    y_soft_trigger_ch3 = [0]*(event_soft_triger)
    y_soft_trigger_ch2 = [0]*(event_soft_triger)
    y_soft_trigger_ch1 = [0]*(event_soft_triger)
    
    item = 0
    
    for event in range(0,events_t):
        if (v_pk_raw_ch1[event] < trigger_ch1 and \
            v_pk_raw_ch2[event] < trigger_ch2 and \
            v_pk_raw_ch3[event] < trigger_ch3 and \
            v_pk_raw_ch1[event] > trigger_sat and \
            v_pk_raw_ch2[event] > trigger_sat and \
            v_pk_raw_ch3[event] > trigger_sat  ):
            
            y_soft_trigger_ch3[item] = y_ch3[event]
            y_soft_trigger_ch2[item] = y_ch2[event]
            y_soft_trigger_ch1[item] = y_ch1[event]
            item = item + 1
    
    print('plot soft-triggered waveforms', file_index,'of',file_num)
#     plot_waveform(time, y_soft_trigger_ch1, xlable="Time(ns)", ylable="Voltage(V)", 
#                   title="ch1 waveform with soft trigger", pdf=False, pic=False)
#     plot_waveform(time, y_soft_trigger_ch2, xlable="Time(ns)", ylable="Voltage(V)", 
#                   title="ch1 waveform with soft trigger", pdf=False, pic=False)
#     plot_waveform(time, y_soft_trigger_ch3, xlable="Time(ns)", ylable="Voltage(V)", 
#                   title="ch1 waveform with soft trigger", pdf=False, pic=False)
    print('plot soft-triggered waveforms done', file_index,'of',file_num)
    
    #### deal with amplitude and charge for cut data#############################
    print('calculate charge and amplitude',  file_index,'of',file_num)
    v_ch1_shift, ampl_ch1 = calculate_voltages(y_soft_trigger_ch1, pedestal_length)
    v_ch2_shift, ampl_ch2 = calculate_voltages(y_soft_trigger_ch2, pedestal_length)
    v_ch3_shift, ampl_ch3 = calculate_voltages(y_soft_trigger_ch3, pedestal_length)

    
    v_ch1_shift = np.array(v_ch1_shift)
    v_ch2_shift = np.array(v_ch2_shift)
    v_ch3_shift = np.array(v_ch3_shift)
    
    q_ch1 = calculate_charge(v_ch1_shift, transCond, time)
    q_ch2 = calculate_charge(v_ch2_shift, transCond, time)
    q_ch3 = calculate_charge(v_ch3_shift, transCond, time)

    
    
    plot_amplitudes(ampl_ch1, ampl_ch2, ampl_ch3, pic=False, pdf=False, num_bins=50, range_ampl=(0.025,0.3))
    plot_charges(q_ch1, q_ch2, q_ch3,  pic=False, pdf=False, num_bins=50, range_q=(0,30))
    #### deal with time#############################
    print('begin to calculate time', file_index,'of',file_num)
#     t_ch1, std_t_ch1, mean_t_ch1 = calculate_time(v_ch1_shift, time, th_cfd=0.5, tdc_bin=0.005, tdc_start = 45)
#     t_ch2, std_t_ch2, mean_t_ch2 = calculate_time(v_ch2_shift, time, th_cfd=0.5, tdc_bin=0.005, tdc_start = 45)
#     t_ch3, std_t_ch3, mean_t_ch3 = calculate_time(v_ch3_shift, time, th_cfd=0.5, tdc_bin=0.005, tdc_start = 45)
    
    t_ch1, std_t_ch1, mean_t_ch1 = calculate_time_cfd1(v_ch1_shift, time, cfd_dly=1, cfd_gain = 0.5, tdc_bin=0.005, 
                                                       tdc_start = 52, title_plot='ch1 cfd', pic=True, pdf = False)
    t_ch2, std_t_ch2, mean_t_ch2 = calculate_time_cfd1(v_ch2_shift, time, cfd_dly=1, cfd_gain = 0.5, tdc_bin=0.005, 
                                                       tdc_start = 52, title_plot='ch2 cfd', pic=True, pdf = False)
    t_ch3, std_t_ch3, mean_t_ch3 = calculate_time_cfd1(v_ch3_shift, time, cfd_dly=1, cfd_gain = 0.5, tdc_bin=0.005, 
                                                       tdc_start = 52, title_plot='ch3 cfd', pic=True, pdf = False)
    print('calculate time done', file_index,'of',file_num)
    
#     plot_times(t_ch1, t_ch2, t_ch3, std_t_ch1, std_t_ch2, std_t_ch3, mean_t_ch1, mean_t_ch2, mean_t_ch3,
#                    pic=False, pdf=False, num_bins=40, range_t=None)
    
    ############### delta t32  ########################
    t32 = [0]*event_soft_triger
    for elem in range(0,event_soft_triger):
        t32[elem] = t_ch3[elem]-t_ch2[elem]
        
    
#     std_t32 = np.std(t32)
#     std_t32 = std_t32 * 1e3  ## to pico second
#     mean_t32 = np.mean(t32)
    
    mean_t32, std_t32 = norm.fit(t32)
    std_t32 = std_t32 * 1e3  ## to pico second
    ############### delta t31  ########################
    t31 = [0]*event_soft_triger
    for elem in range(0,event_soft_triger):
        t31[elem] = t_ch3[elem]-t_ch1[elem]
        
    
#     std_t31 = np.std(t31)
#     std_t31 = std_t31 * 1e3  ## to pico second
#     mean_t31 = np.mean(t31)
    
    mean_t31, std_t31 = norm.fit(t31)
    std_t31 = std_t31 * 1e3  ## to pico second
    
    ############### delta t21  ########################
    t21 = [0]*event_soft_triger
    for elem in range(0,event_soft_triger):
        t21[elem] = t_ch2[elem]-t_ch1[elem]
        
    
#     std_t21 = np.std(t21)
#     std_t21 = std_t21 * 1e3  ## to pico second
#     mean_t21 = np.mean(t21)
    
    mean_t21, std_t21 = norm.fit(t21)
    std_t21 = std_t21 * 1e3  ## to pico second
    
    ############### delta t123  ########################
    t123 = [0]*event_soft_triger
    for elem in range(0,event_soft_triger):
        t123[elem] = (t_ch1[elem] + t_ch3[elem])/2 - t_ch2[elem]
        
    
#     std_t123 = np.std(t123)
#     std_t123 = std_t123 * 1e3  ## to pico second
#     mean_t123 = np.mean(t123)
    
    mean_t123, std_t123 = norm.fit(t123)
    std_t123 = std_t123 * 1e3  ## to pico second
    
    return (ampl_ch1, ampl_ch2, ampl_ch3, q_ch1, q_ch2, q_ch3, 
    t_ch1, std_t_ch1, mean_t_ch1, t_ch2, std_t_ch2, mean_t_ch2,
    t_ch3, std_t_ch3, mean_t_ch3, t32, std_t32, mean_t32, 
    t31, std_t31, mean_t31, t21, std_t21, mean_t21, t123, std_t123, mean_t123)

def analyze_single_charge(hdf5_in='test.hdf5', path='../test/', charge_size='0 fC', DAC='000', tdc_bin=5e-3, transCond = 4.4e3, tdc_start_trigger=9.8,
                tdc_start_pa=18, tdc_le_start=20, tdc_te_start=30):
    file = path + hdf5_in
    data, attrs = extract_dataset(file)
    events_t = attrs['readout_size']
    npoints = attrs['nPt']
    x = np.arange(npoints)
    dt = attrs['dt']
    
    filelist = []
    filelist_to_use = [hdf5_in]
    for apath in filelist_to_use:
        if apath.find(".hdf5") > -1:
            filelist.append(os.path.join(path,apath))
    y_ch1 = []
    y_ch2 = []
    y_ch3 = []
    y_ch4 = []
    for item in tqdm(filelist):
        #data, attrs = etroc0_benchtest.extract_dataset(item)
        data, attrs = extract_dataset(item)
        for event in range(len(data[0])):
            y_ch1.append(data[0][event])
        for event in range(len(data[1])):
            y_ch2.append(data[1][event])
        for event in range(len(data[2])):
            y_ch3.append(data[2][event])
        for event in range(len(data[3])):
            y_ch4.append(data[3][event])
    
    events_ch1 = len(y_ch1)
    events_ch2 = len(y_ch2)
    events_ch3 = len(y_ch3)
    events_ch4 = len(y_ch4)
    events_t = events_ch1
    time = x*dt*1e9
    
    print('npoints at', charge_size,'/',DAC, 'is', npoints)
    print('events_t is',events_t)
    
    ######################################## trigger  ##########################################
    vth_trigger = 0.5*(np.max(y_ch3[1]) - np.min(y_ch3[1]))   # V
    print('threshold of trigger is', vth_trigger)
    tdc_start_trigger = tdc_start_trigger  # ns
    
    t_trigger = calculate_time_qinj_trigger(
        y_ch3, time, vth_trigger, tdc_bin, tdc_start_trigger)
    
    mean_t_trigger, std_t_trigger = norm.fit(t_trigger)
    
    print('trigger mean is',mean_t_trigger, 'ns')
    print('trigger std is',std_t_trigger, 'ns')
    print('**************************************************************************************')
    
    ######################################## preamp  ###########################################
    vth_pa = -0.03   # V
    print('threshold of PA is', vth_pa)
    tdc_start_pa = tdc_start_pa  # ns
    t_pa, std_t_pa, mean_t_pa = calculate_time_qinj_amp(
        y_ch1, time, vth_pa, tdc_bin, tdc_start_pa)
    
    mean_t_pa, std_t_pa = norm.fit(t_pa)
    
    print('PA mean is',mean_t_pa, 'ns')
    print('PA std is',std_t_pa, 'ns')
    print('**************************************************************************************')
    
    ######################################## discriminator  ##########################################
    vth_discri = 0.28   # V
    tdc_leading_start = tdc_le_start  # ns
    tdc_trialing_start = tdc_te_start  # ns
    
    t_discri_le, t_discri_te = calculate_time_qinj_discri(
        y_ch2, time, vth_discri, tdc_bin, tdc_leading_start, tdc_trialing_start)
    
    mean_t_discri_le, std_t_discri_le = norm.fit(t_discri_le)
    
    mean_t_discri_te, std_t_discri_te = norm.fit(t_discri_te)
    
    print('discriminator leading mean is',mean_t_discri_le, 'ns')
    print('discriminator leading std is',std_t_discri_le, 'ns')
    print('discriminator trailing mean is',mean_t_discri_te, 'ns')
    print('discriminator trailing std is',std_t_discri_te, 'ns')
    print('**************************************************************************************')
    
    ####################################### discriminator time with trigger removed  ################
    t_discri_le_notrig = [0]*events_t
    for index in range(0,events_t):
        t_discri_le_notrig[index] = t_discri_le[index] - t_trigger[index]
    
    t_discri_te_notrig = [0]*events_t
    for index in range(0,events_t):
        t_discri_te_notrig[index] = t_discri_te[index] - t_trigger[index]    
        
    mean_t_discri_le_notrig, std_t_discri_le_notrig = norm.fit(t_discri_le_notrig)  
    mean_t_discri_te_notrig, std_t_discri_te_notrig = norm.fit(t_discri_te_notrig)
    
    print('discriminator t_le mean (trigger removed) is', mean_t_discri_le_notrig, 'ns')
    print('discriminator t_le std (trigger removed) is', std_t_discri_le_notrig, 'ns')
    print('discriminator t_te mean (trigger removed) is', mean_t_discri_te_notrig, 'ns')
    print('discriminator t_te std (trigger removed) is', std_t_discri_te_notrig, 'ns')
    print('**************************************************************************************')
    
    #### deal with amplitude and charge for cut data#############################
    print('calculate charge and amplitude',  charge_size,'/',DAC)
    pedestal_length = int(tdc_start_pa/(dt*1e9))
    v_ch1_shift, ampl_ch1 = calculate_voltages(y_ch1, pedestal_length=pedestal_length)
    v_ch1_shift = np.array(v_ch1_shift)
    q_ch1 = calculate_charge(v_ch1_shift, transCond, time)
    
    mean_ampl, std_ampl = norm.fit(ampl_ch1) 
    mean_q, std_q = norm.fit(q_ch1) 

    
    return t_trigger, mean_t_trigger, std_t_trigger, t_pa, mean_t_pa, std_t_pa, t_discri_le, mean_t_discri_le, std_t_discri_le, t_discri_te, mean_t_discri_te, std_t_discri_te, t_discri_le_notrig, mean_t_discri_le_notrig, std_t_discri_le_notrig, t_discri_te_notrig, mean_t_discri_te_notrig, std_t_discri_te_notrig, ampl_ch1, mean_ampl, std_ampl, q_ch1, mean_q, std_q

def plot_distribution_stack(list_in, file_num, filelist, num_bins = 20, range_default = None, xaxis='Charge(fC)',
                           plot_name='Charge Distribution of Ch2'):
    fig, ax1 = plt.subplots(dpi=200)
    for item in range(file_num):
        charge_name = filelist[item].split('_')[1]
        ax1.hist(list_in[item], num_bins, range=range_default, density=False,
                 label='%s, events:%d'%(charge_name,len(list_in[item])))
    ax1.legend(fontsize = 'x-small')
    ax1.grid()
    ax1.set(xlabel=xaxis, ylabel='Occurance',
           title=plot_name)
    plt.show()
    plt.close(fig)


def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gaus_fit(to_fit, num_bins=50):
    mean = np.mean(to_fit)
    sigma = np.std(to_fit, ddof=1)
    bins, edges = np.histogram(to_fit, num_bins, density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    popt, pcov = curve_fit(gaus,centers,bins,p0=[1,mean,sigma])
    mean = popt[1]
    sigma = popt[2]
    return mean,sigma, popt, pcov

def plot_distribution_time(list_in, file_item, num_bins= 20, range_default = None, xaxis = 'Time Resolution(ns)',
                          ylable = 'Occurrence', title = 'r$\delta$', pic = False, pdf = True):
    entries = len(list_in)
    mu, std = norm.fit(list_in)
    fig, ax4= plt.subplots(dpi=200)
    n,bins,patches=ax4.hist(list_in, bins=num_bins, range=range_default, density=False, 
                            label = 'entries: %d\nstd:%f\nmean:%f'%(entries, std, mu))
    xmin = np.min(list_in)
    xmax = np.max(list_in)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax4.plot(x, np.max(n)*p/np.max(p), 'g', linewidth=2)
    ax4.grid()
    ax4.legend()
    ax4.set(xlabel='Time Resolution(ns)', ylabel='Occurrence',
               title=title)
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)

def plot_distribution_time_Lindsey(list_in, file_item, num_bins= 20, range_default = None, xaxis = 'Time Resolution(ns)',
                          ylable = 'Occurrence', title = 'r$\delta$', pic = False, pdf = True):
    
    to_fit = list_in
    mean = np.mean(to_fit)
    sigma = np.std(to_fit, ddof=1)
    bins, edges = np.histogram(to_fit, num_bins, density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    popt, pcov = curve_fit(gaus,centers,bins,p0=[1,mean,sigma])
    mean = popt[1]
    sigma = popt[2]
    
    entries=len(to_fit)
    fig, ax4= plt.subplots(dpi=200)
    ax4.hist(to_fit, bins=num_bins, range=range_default, density=False, 
                            label = 'entries: %d\nstd: %.4f\nmean: %.2f'%(entries, abs(sigma), mean))
    ax4.plot(centers, gaus(centers,popt[0], popt[1], popt[2]), 'g', linewidth=2)
    ax4.grid()
    ax4.legend()
    ax4.set(xlabel='Time Resolution(ns)', ylabel='Occurrence',
               title=title)
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)

def plot_distribution_charge(list_in, file_item, num_bins= 50, range_default = None, pic = False, pdf = False):
    entries = len(list_in)
    scale = np.std(list_in, ddof=1)
    loc = np.mean(list_in)
    norm = list_in.size
    bins, edges = np.histogram(list_in, num_bins, density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    popt, pcov = curve_fit(moyal, centers, bins, p0=[norm, loc,scale])
    norm = popt[0]
    loc = popt[1]
    scale = popt[2]
    
    fig, ax4= plt.subplots(dpi=100)
    ax4.hist(list_in, bins=num_bins, range=range_default, density=False, 
                            label='MPV = %.2f fC\n#event = %d\n#bin = %d'%(loc,list_in.size,num_bins))
    ax4.plot(centers, moyal(centers, popt[0], popt[1], popt[2]),'g', linewidth=2)
    ax4.grid()
    ax4.legend()
    ax4.set(xlabel='Charge(fC)', ylabel='Occurrence',
               title='Charge Distribution of %s'%file_item)
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    return loc

def plot_distribution_ampl(list_in, file_item, num_bins= 50, range_default = None, pic = False, pdf = False):
    entries = len(list_in)
    scale = np.std(list_in, ddof=1)
    loc = np.mean(list_in)
    norm = len(list_in)
    bins, edges = np.histogram(list_in, num_bins, density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    popt, pcov = curve_fit(moyal, centers, bins, p0=[norm, loc,scale])
    norm = popt[0]
    loc = popt[1]
    scale = popt[2]
    
    fig, ax4= plt.subplots(dpi=100)
    ax4.hist(list_in, bins=num_bins, range=range_default, density=False, 
                            label='MPV = %.3f V\n#event = %d\n#bin = %d'%(loc,entries,num_bins))
    ax4.plot(centers, moyal(centers, popt[0], popt[1], popt[2]),'g', linewidth=2)
    ax4.grid()
    ax4.legend()
    ax4.set(xlabel='Amplitude(V)', ylabel='Occurrence',
               title='Amplitude Distribution of %s'%file_item)
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    return loc

def plot_TWCorrection(toa, tot, popt, t_ch4, toatottitle = 'TOA vs TOT', distriTitle = 'ch4 time', num_bins=50, pic=False, pdf=False):
    fig,ax1 = plt.subplots(dpi=200)
    ax1.plot(tot, toa, 'b.', label='data')
    
    #print fit
    xmin, xmax = plt.xlim()
    xtot = np.linspace(xmin, xmax, 1000)
    ax1.plot(xtot, func1(xtot, *popt), 'g', label='fit TOA vs TOT: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    ax1.set(xlabel='TOT(ns)', ylabel='TOA(ns)',  title='TOA vs TOT')
    ax1.grid()
    ax1.legend()
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    
    plot_distribution_time(t_ch4, 'file_item', num_bins= num_bins, range_default = None, xaxis = 'Time Resolution(ns)',
                          ylable = 'Occurrence', title = distriTitle , pic = pic, pdf = pdf)