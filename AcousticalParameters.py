import numpy as np
from tkinter import sys
from scipy import signal
from scipy.ndimage import median_filter
import pandas as pd






# ---------------------------------- GUI FUNCTIONS ----------------------------------------------







# Acoustic parameters are calculated for a mono IR. If "ter=1" it filters by third octave band and if "ter=0" it filters by octave band. "w" is MFF size window in ms. 
def MonoParam(IR, Fs, ter, smooth, w):

    IRcut = IRm_cut(IR, Fs)
    
# Filter RIR 
    mono, cen = FiltMono(IRcut, Fs, ter)
    mono = ETCnorm(mono)
    
# Obtain Energy Time Curve and smoothed curve
    etc = ETCnorm(np.array([IRcut]))
    etc = etc[0]
    ETC = 10 * np.log10(etc + sys.float_info.epsilon)
        
    ir_tr = []
    ir_sm = []
    lb = []
      
    signals = np.append(mono, np.array([etc]), axis=0)
    
    for i, x in enumerate(signals):

            punto_cruce, c = Lund(x, Fs)
            lb.append(punto_cruce)
            ir_tr.append(x[:punto_cruce])
            p = x.size-punto_cruce
        
# Smoothing (Schroeder or MMF)
            if smooth == 1:
                mmf = MovMed(ir_tr[i], w, Fs, p)
                ir_sm.append(mmf)
            
            elif smooth == 0:
               sch = Sch(ir_tr[i], p)
               ir_sm.append(sch)

    ir_tr.pop(-1)
    smooth_ir = ir_sm.pop(-1)
    
    Tt, EDTt = Tt_EDTt(ir_tr, ir_sm, Fs)
    EDT, T20, T30 = RTparam(ir_sm, Fs)
    C50, C80 = Clarityparam(mono, Fs, lb)
    
    MonParam = {'Tt': Tt,'EDTt': EDTt,'C50': C50,'C80': C80,'EDT': EDT,'T20': T20,'T30': T30,'ETC': ETC,'smooth': smooth_ir}
    
    return MonParam

# Acoustic parameters are calculated for a stereo splited IR. If "ter=1" it filters by third octave band and if "ter=0" it filters by octave band. "w" is MMF size window in ms.
def StereoSplitParam(IRl, IRr, Fs, ter, smooth, w):
 
    IRlcut, IRrcut = IRs_cut(IRl, IRr, Fs)
    IRl_filt, IRr_filt, cen = FiltStereo(IRlcut, IRrcut, Fs, ter)
    
# Add IACCe parameter to results and append curves of both channels    
    IACCe = IACCearly(IRl_filt, IRr_filt, Fs)
    
    paramIRl = MonoParam(IRl, Fs, ter, smooth, w)
    paramIRr = MonoParam(IRr, Fs, ter, smooth, w)   
    sParam = paramIRl.copy() 
    sParam['IACCe'] = IACCe
    sParam['ETC'] = [paramIRl['ETC'], paramIRr['ETC']]
    sParam['smooth'] = [paramIRl['smooth'], paramIRr['smooth']]
    
    return sParam

# Acoustic parameters are calculated for a stereo IR. If "ter=1" it filters by third octave band and if "ter=0" it filters by octave band. "w" is MMF size window in ms.
def StereoParam(IR, Fs, ter, smooth, w):
    
    IRr, IRl = SepStereo(IR)
    IRlcut, IRrcut = IRs_cut(IRl, IRr, Fs)
    IRl_filt, IRr_filt, cen = FiltStereo(IRlcut, IRrcut, Fs, ter)

# Add IACCe parameter to results and append curves of both channels    
    IACCe = IACCearly(IRl_filt, IRr_filt, Fs)
    
    paramIRl = MonoParam(IRl, Fs, ter, smooth, w)
    paramIRr = MonoParam(IRr, Fs, ter, smooth, w)   
    sParam = paramIRl.copy() 
    sParam['IACCe'] = IACCe
    sParam['ETC'] = [paramIRl['ETC'], paramIRr['ETC']]
    sParam['smooth'] = [paramIRl['smooth'], paramIRr['smooth']]
    
    return sParam

#  A dataframe is created to store the acoustic parameters and dump them in the GUI. If "ter=1" it filters by third octave band and if "ter=0" it filters by octave band.
def Dataframe(Param, ter):

    data = Param.copy()
    
    if ter:
        freqs = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160','200', '250', '315', '400', '500', '630', '800', '1k','1.3k','1.6k', '2k', '2.5k', '3.2k', '4k', '5k', '6.3k', '8k', '10k', '12.5k', '16k', '20k']
    else:
        freqs = ['31.5', '63', '125', '250', '500', '1000', '2000', '4000', '8000', '16000']
    
# Create DataFrame   
    del data['ETC']
    del data['smooth']
    
    df = pd.DataFrame.from_dict(data, 
                                orient='index',
                                columns=freqs)
   
    return df








# -------------------------------------- FUNCTIONS ---------------------------------------------------------------








# A stereo audio file is separated into two channels: L and R.
def SepStereo(IRstereo):
    
    IRl = IRstereo[:,0]
    IRr = IRstereo[:,1]
    
    return IRr, IRl

# The IACC early parameter is calculated according to the ISO-3382 standard, for a stereo IR.
def IACCearly(IRl, IRr, Fs):

# The variable is initialized.     
    IACCe = []
    
    for ir_L, ir_R in zip(IRl, IRr):
        t80 = np.int64(0.08*Fs)
        I = np.correlate(ir_L[0:t80], ir_R[0:t80], 'full')/(np.sqrt(np.sum(ir_L[0:t80]**2)*np.sum(ir_R[0:t80]**2)))
        iacce = np.max(np.abs(I))
        
        IACCe.append(iacce)
        
    IACCe = np.round(IACCe, 2)
        
    return IACCe

# The IR is cut, from the first maximum that it presents. If the lenght of the resulting IR is over 15 seconds, it is shortened to 15 seconds to avoid overly large data to be processed.
def IRm_cut(IR, Fs):
    
    IRmax = np.where(abs(IR) == np.max(abs(IR)))[0][0]
    IRcut = IR[(IRmax)+5:]
    
    if len(IRcut) / Fs > 15: 
        IRcut = IRcut[:int(15 * Fs)]    
    return IRcut  

# The stereo IR is cut, using the IRm_cut function on each.
# In addition, the lengths of the signals are equalized.
def IRs_cut(IRl, IRr, Fs):
  
    IRlcut = IRm_cut(IRl, Fs)
    IRrcut = IRm_cut(IRr, Fs)

    if IRrcut.size > IRlcut.size:
        IRrcut[:IRlcut.size]    
    else:
        IRlcut[:IRrcut.size]
    
    return IRlcut, IRrcut

#  The ETC of an IR is calculated and normalized.
def ETCnorm(IR):
    
    ETC = np.zeros(IR.shape)  
    
    for i, y in enumerate(IR):
        E = np.abs(signal.hilbert(y))**2
        ETC[i] = E/np.max(E)
        
    return ETC

# The signal is filtered using a moving average filter, with a window size "w" chosen by the user.
def MovMed(IR, w, Fs, p):

    v = int(w*Fs/1000)
    if v % 2 == 0:
        v +=1    
    filt = median_filter(IR,v)   
    if p:
        filt = np.concatenate((filt, np.zeros(p)))
    with np.errstate(divide='ignore', invalid='ignore'):
        filt = 10*np.log10(filt / np.max(filt))
    return filt

# Calculate the inverse Schroeder integral of an IR.
def Sch(IR, p):

    sch = np.cumsum(IR[::-1])[::-1]
    
    if p:
        sch = np.concatenate((sch, np.zeros(p)))

    with np.errstate(divide='ignore', invalid='ignore'):
        sch_db = 10.0 * np.log10(sch / np.max(sch))
    
    return sch_db


# An IR is filtered, using butterworth bandpass filters. If "ter=1" it filters by third octave band and if "ter=0" it filters by octave band.
def FiltMono(IR, Fs, ter):
    
    W = np.flip(IR) 
    G = 10**(3/10)
    fil = []

    if ter:
        Fc_center = np.array([25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 
                              250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                              2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
                              12500, 16000, 20000])
        fmin = G ** (-1/6)
        fmax = G ** (1/6)
    else:
        Fc_center = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000,
                              8000, 16000])
        fmin = G ** (-1/2)
        fmax = G ** (1/2)
            
    for j, fc in enumerate(Fc_center):
        
# Define the upper and lower limits of the frequency band.
        sup = fmax*fc/(0.5*Fs)         
        if sup >= 1:
            sup = 0.999999            
        inf = fmin * fc / (0.5*Fs) 

# Second order IIR Butterworth filter is applied.
        sos = signal.butter(N=2, Wn=np.array([inf, sup]), 
                            btype='bandpass',output='sos')        
        filt = signal.sosfilt(sos, W)
        fil.append(filt) 
        fil[j] = np.flip(fil[j])   
    IRfilt = np.array(fil)
    
# Cut the last 5% of the signal to minimize the border effect.
    IRfilt = IRfilt[:int(len(fil[1])*0.95)]
    
    return IRfilt, Fc_center

# Calculate the parameters EDTt and Tt of a filtered IR.
def Tt_EDTt(IR, y, Fs):

# Variables are initialized    
    EDTt = []
    Tt = []
    
    for i, ir in enumerate(IR):
        
# Remove the first 5 ms
        ir = ir[int(5e-3 * Fs):]  
        
# Find the index of the upper limit of the interval that contains 99% of the energy.         
        index = np.where(np.cumsum(ir ** 2) <= 0.99 * np.sum(ir ** 2))[0][-1]
        t_t = index/Fs
        Tt.append(t_t)
                
# Filter the impulse with the moving median filter in order to calculate the parameters.        
        ir2 = y[i]
        ir2 = ir2[:index]  
        t_Tt = np.arange(0, index/Fs, 1/Fs)
        
        if len(t_Tt) > index:
            t_Tt = t_Tt[:index]
        
# Calculate minimum squares.           
        A = np.vstack([t_Tt, np.ones(len(t_Tt))]).T
        m, c = np.linalg.lstsq(A, ir2, rcond=-1)[0]        
        edt_t = -60/m
        EDTt.append(edt_t)        
    EDTt = np.round(EDTt, 2)
    Tt = np.round(Tt, 2)
    
    return Tt, EDTt

# Stereo IR filtering, using butterworth bandpass filters. If "ter=1" it filters by third octave band and if "ter=0" it filters by octave band.
def FiltStereo(IRl, IRr, Fs, ter):
  
    IRl_filt = FiltMono(IRl, Fs, ter)
    IRr_filt = FiltMono(IRr, Fs, ter)
    
    IRl_filt = IRl_filt[0]
    IRr_filt = IRr_filt[0]
    Fc_center = IRl_filt[1]
                    
    return  IRl_filt, IRr_filt, Fc_center


# The integration limits of the inverse Schroeder integral are found according to Lundeby's method.
def Lund(IR, Fs):
   
    N = IR.size
    energy = IR
    med = np.zeros(int(N/(Fs*0.01)))
    eje_tiempo = np.zeros(int(N/(Fs*0.01)))
    
# Divide in sections and calculate the mean.    
    t = np.floor(N/(Fs*0.01)).astype('int')
    v = np.floor(N/t).astype('int')   
    for i in range(0, t):
        med[i] = np.mean(energy[i * v:(i + 1) * v])
        eje_tiempo[i] = np.ceil(v/2).astype('int') + (i*v)
        
# Calculate noise level of the last 10% of the signal.    
    rms_dB = 10 * np.log10(np.sum(energy[round(0.9 * N):]) / (0.1 * N) / max(energy))
    meddB = 10 * np.log10(med / max(energy))

# The linear regression of the 0dB interval and the mean closest to the noise + 10dB is sought.   
    try:
        r = int(max(np.argwhere(meddB > rms_dB + 10)))
           
        if np.any(meddB[0:r] < rms_dB+10):
            r = min(min(np.where(meddB[0:r] < rms_dB + 10)))
        if np.all(r==0) or r<10:
            r=10
    except:
        r = 10

# Least squares.       
    A = np.vstack([eje_tiempo[0:r], np.ones(len(eje_tiempo[0:r]))]).T
    m, c = np.linalg.lstsq(A, meddB[0:r], rcond=-1)[0]
    cruce = int((rms_dB-c)/m)
    
# Insufficient SNR.    
    if rms_dB > -20:        
        punto = len(energy)
        C = None        
    else:
        error = 1
        INTMAX = 50
        veces = 1               
        while error > 0.0004 and veces <= INTMAX:
            
# Calculates new time intervals for the mean with approximately p steps for each 10 dB.            
            p = 10
            
# Number of samples for the decay slope of 10 dB.            
            delta = int(abs(10/m)) 
            
# Interval over which the mean is calculated.           
            v = np.floor(delta/p).astype('int') 
            t = int(np.floor(len(energy[:int(cruce-delta)])/v))            
            if t < 2:
                t = 2
                
            elif np.all(t == 0):
                t = 2
            media = np.zeros(t)
            eje_tiempo = np.zeros(t)
            
            for i in range(0, t):
                media[i] = np.mean(energy[i*v:(i + 1) * v])
                eje_tiempo[i] = np.ceil(v / 2) + (i * v).astype('int')
                
            mediadB = 10 * np.log10(media / max(energy))
            A = np.vstack([eje_tiempo, np.ones(len(eje_tiempo))]).T
            m, c = np.linalg.lstsq(A, mediadB, rcond=-1)[0]

# New noise average level, starting from the point of the decay curve, 10 dB below the intersection.           
            noise = energy[int(abs(cruce + delta)):]
            
            if len(noise) < round(0.1 * len(energy)):
                noise = energy[round(0.9 * len(energy)):]
                
            rms_dB = 10 * np.log10(sum(noise)/ len(noise) / max(energy))

# New intersection index           
            error = abs(cruce - (rms_dB - c) / m) / cruce
            cruce = round((rms_dB - c) / m)
            veces += 1
                   
# Output validation            
    if cruce > N:
        punto = N
    else:
        punto = int(cruce)       
    C = max(energy) * 10 ** (c / 10) * np.exp(m/10/np.log10(np.exp(1))*cruce) / (
        -m / 10 / np.log10(np.exp(1)))
        
    return punto, C


# The EDT, T20 and T30 parameters are calculated from a filtered IR.
def RTparam(IRfilt, Fs):

# Variables are initialized and created time array.    
    results = {'EDT': [],
               'T20': [],
               'T30': []}   
    t = np.arange(0, len(IRfilt[0])/Fs, 1/Fs)
    
    for ir in IRfilt:
    
# Look for maximum values and their indexes.       
        i_max = np.where(ir == max(ir))[0][0]           
        y = ir[int(i_max):]
        y_max = max(y)
        
# Get the indexes where the level of the signal is between the defined limits.        
        i_edt = np.where((y <= y_max - 1) & (y > (y_max - 10)))
        i_20 = np.where((y <= y_max - 5) & (y > (y_max - 25)))    
        i_30 = np.where((y <= y_max - 5) & (y > (y_max - 35)))        
        indexes = [i_edt, i_20, i_30]
        
        for i, key in zip(indexes, results):
            
            t_tr = np.vstack([t[i], np.ones(len(t[i]))]).T
            y_tr = y[i]
            
# Calculate linear regression to extrapolate reverberation time and append results to dictionary.            
            m, c = np.linalg.lstsq(t_tr, y_tr, rcond=-1)[0]
            result = -60/m
            results[key].append(result)
        
    EDT = np.round(results['EDT'], 2)
    T20 = np.round(results['T20'], 2)
    T30 = np.round(results['T30'], 2)
    
    return EDT, T20, T30

# Parameters C50 and C80 are calculated from a filtered IR. "lb" are the limits of integration.
def Clarityparam(IRfilt, Fs, lb):
    
# Variables are initialized    
    C50 = []
    C80 = []   
    
    for y, z in zip(IRfilt, lb):
        t50 = np.int64(0.05*Fs)  # Index of signal value at 50 ms.
        t80 = np.int64(0.08*Fs)  # Index of signal value at 80 ms.
        
        y50_num = y[0:t50]
        y50_den = y[t50:z]
        y80_num = y[0:t80]
        y80_den = y[t80:z]
    
        c50 = 10*np.log10(np.sum(y50_num) / np.sum(y50_den))
        c80 = 10*np.log10(np.sum(y80_num) / np.sum(y80_den))
        C50.append(c50)
        C80.append(c80)
        
    C50 = np.round(C50,2)
    C80 = np.round(C80,2)
    
    return C50, C80




