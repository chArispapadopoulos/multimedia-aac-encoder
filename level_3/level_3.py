#lvl 3 
import numpy as np
import soundfile as sf
from scipy.fft import dct, fft
from scipy.signal import lfilter
from scipy.signal.windows import kaiser
from scipy.linalg import toeplitz, solve
from numpy.polynomial import Polynomial
import scipy.io
import os
import huff_utils
import matplotlib.pyplot as plt

#GLOBAL VARIABLE FOR THE PLOTS
GLOBAL_PLOT_DATA = []

#EPS value computation 
EPS = np.finfo(float).eps
#Helpers for the demo and the results
def bitlen(x):
    if x is None: return 0
    if isinstance(x, str): return len(x)
    arr = np.array(x).flatten()
    if arr.size == 1 and isinstance(arr[0], str): return len(arr[0])
    return int(arr.size)

def safe_snr_db(signal_pow, noise_pow, atol=1e-20):
    if np.isfinite(noise_pow) and (noise_pow <= atol or np.isclose(noise_pow, 0.0, atol=atol)):
        return np.inf
    return 10.0 * np.log10(signal_pow / noise_pow)

def print_snr(label, snr):
    if np.isinf(snr): print(f"{label}: inf dB")
    else: print(f"{label}: {snr:.2f} dB")

#Helper functions for lvl 1 & 2
def get_kbd_window(N, alpha=4.0):
    M = N // 2
    w = kaiser(M + 1, beta=np.pi * alpha)
    w_cumsum = np.cumsum(w)
    denom = w_cumsum[-1]
    win_left = np.sqrt(w_cumsum[:M] / denom)
    win_right = win_left[::-1]
    return np.concatenate((win_left, win_right))

def derive_window(win_type, N):
    if win_type == "KBD": return get_kbd_window(N, alpha=6.0 if N == 2048 else 4.0)
    n = np.arange(N)
    return np.sin(np.pi / N * (n + 0.5))

def mdct4(x):
    N = x.shape[0] // 2; n4 = N // 2
    a, b, c, d = x[0:n4], x[n4:2*n4], x[2*n4:3*n4], x[3*n4:4*n4]
    folded = np.concatenate((-c[::-1] - d, a - b[::-1]))
    return dct(folded, type=4, norm='ortho')

def imdct4(X):
    N = X.shape[0]; n2 = N // 2
    y_folded = dct(X, type=4, norm='ortho')
    y0, y1 = y_folded[0:n2], y_folded[n2:N]
    return np.concatenate((y1, -y1[::-1], -y0[::-1], -y0))

# LEVEL 1 & 2
def SSC(frame_T, next_frame_T, prev_frame_type):
    b = np.array([0.7548, -0.7548])
    a = np.array([1.0, -0.5095])
    channels = next_frame_T.shape[1]
    
    is_next_esh = [False, False]
    for ch in range(channels):
        x_filt = lfilter(b, a, next_frame_T[:, ch])
        s2 = [np.sum(x_filt[448 + i*128: 448 + (i+1)*128]**2) for i in range(8)]
        for l in range(1, 8):
            avg_prev = sum(s2[:l]) / l
            if s2[l] > 1e-3 and s2[l] > 10 * avg_prev: 
                is_next_esh[ch] = True
                break
                
    def get_channel_type(prev_type, next_is_esh):
        if prev_type == "OLS": return "LSS" if next_is_esh else "OLS"
        elif prev_type == "LSS": return "ESH"
        elif prev_type == "ESH": return "ESH" if next_is_esh else "LPS"
        elif prev_type == "LPS": return "OLS"
        return "OLS"
        
    type_ch0 = get_channel_type(prev_frame_type, is_next_esh[0])
    type_ch1 = get_channel_type(prev_frame_type, is_next_esh[1])
    
    table_1 = {
        ("OLS", "OLS"): "OLS", ("OLS", "LSS"): "LSS", ("OLS", "ESH"): "ESH", ("OLS", "LPS"): "LPS",
        ("LSS", "OLS"): "LSS", ("LSS", "LSS"): "LSS", ("LSS", "ESH"): "ESH", ("LSS", "LPS"): "ESH",
        ("ESH", "OLS"): "ESH", ("ESH", "LSS"): "ESH", ("ESH", "ESH"): "ESH", ("ESH", "LPS"): "ESH",
        ("LPS", "OLS"): "LPS", ("LPS", "LSS"): "ESH", ("LPS", "ESH"): "ESH", ("LPS", "LPS"): "LPS"
    }
    
    return table_1[(type_ch0, type_ch1)]

def filter_bank(frame_T, frame_type, win_type):
    channels = frame_T.shape[1]; frame_F = np.zeros((1024, channels))
    win_long = derive_window(win_type, 2048); win_short = derive_window(win_type, 256)
    for ch in range(channels):
        x = frame_T[:, ch]
        if frame_type == "OLS": frame_F[:, ch] = mdct4(x * win_long)
        elif frame_type == "LSS":
            w = np.zeros(2048); w[:1024] = win_long[:1024]; w[1024:1472] = 1.0; w[1472:1600] = win_short[128:]
            frame_F[:, ch] = mdct4(x * w)
        elif frame_type == "LPS":
            w = np.zeros(2048); w[448:576] = win_short[:128]; w[576:1024] = 1.0; w[1024:] = win_long[1024:]
            frame_F[:, ch] = mdct4(x * w)
        elif frame_type == "ESH":
            coeffs = [mdct4(x[448 + k*128: 448 + k*128 + 256] * win_short) for k in range(8)]
            frame_F[:, ch] = np.concatenate(coeffs)
    return frame_F

def i_filter_bank(frame_F, frame_type, win_type):
    channels = frame_F.shape[1]; frame_T = np.zeros((2048, channels))
    win_long = derive_window(win_type, 2048); win_short = derive_window(win_type, 256)
    for ch in range(channels):
        spec = frame_F[:, ch]
        if frame_type == "OLS": frame_T[:, ch] = imdct4(spec) * win_long
        elif frame_type == "LSS":
            w = np.zeros(2048); w[:1024] = win_long[:1024]; w[1024:1472] = 1.0; w[1472:1600] = win_short[128:]
            frame_T[:, ch] = imdct4(spec) * w
        elif frame_type == "LPS":
            w = np.zeros(2048); w[448:576] = win_short[:128]; w[576:1024] = 1.0; w[1024:] = win_long[1024:]
            frame_T[:, ch] = imdct4(spec) * w
        elif frame_type == "ESH":
            temp = np.zeros(2048); specs = np.split(spec, 8)
            for k in range(8): temp[448 + k*128: 448 + k*128 + 256] += imdct4(specs[k]) * win_short
            frame_T[:, ch] = temp
    return frame_T

def load_tns_tables(mat_filename='TableB219.mat'):
    mat = scipy.io.loadmat(mat_filename)
    return mat['B219a'], mat['B219b']
B219a, B219b = load_tns_tables('TableB219.mat')

def tns(frame_F_in, frame_type):
    is_esh = (frame_type == "ESH")
    if is_esh:
        X_subframes = np.transpose(frame_F_in.reshape(128, 8, order='F'))
        num_subframes = 8; N = 128; bands_info = B219b
    else:
        X_subframes = frame_F_in.reshape(1, 1024)
        num_subframes = 1; N = 1024; bands_info = B219a
    starts = bands_info[:, 1].astype(int); ends = bands_info[:, 2].astype(int) + 1
    if (not is_esh) and starts[0] == 1: starts -= 1; ends -= 1
    starts = np.clip(starts, 0, N); ends = np.clip(ends, 0, N)
    X_out = np.zeros_like(X_subframes); tns_coeffs_out = np.zeros((4, num_subframes))
    for i in range(num_subframes):
        X_k = X_subframes[i, :]; Sw = np.zeros(N)
        for b_idx in range(len(bands_info)):
            s, e = starts[b_idx], ends[b_idx]
            if s >= e: continue
            Sw[s:e] = np.sqrt(np.sum(X_k[s:e]**2) + 1e-20)
        for k in range(N - 2, -1, -1): Sw[k] = (Sw[k] + Sw[k+1]) / 2.0
        for k in range(1, N ): Sw[k] = (Sw[k] + Sw[k-1]) / 2.0
        Sw[Sw < 1e-20] = 1.0; Xw = X_k / Sw
        corr = np.correlate(Xw, Xw, mode='full'); mid = len(Xw) - 1; r = corr[mid: mid + 5]
        if r[0] < 1e-10 or np.isnan(r).any(): coeffs = np.zeros(4)
        else:
            try: coeffs = solve(toeplitz(r[:-1]), r[1:])
            except Exception: coeffs = np.zeros(4)
        coeffs = np.clip(np.round(coeffs / 0.1), -8, 7) * 0.1
        roots = Polynomial(np.concatenate((-coeffs[::-1], [1.0]))).roots()
        if np.any(np.abs(roots) >= 1.0): coeffs = np.zeros(4)
        tns_coeffs_out[:, i] = coeffs
        X_out[i, :] = lfilter(np.concatenate(([1.0], -coeffs)), [1.0], X_k)
    if is_esh: return X_out.T, tns_coeffs_out
    return X_out.reshape(1024, 1), tns_coeffs_out.reshape(4, 1)

def i_tns(frame_F_in, frame_type, tns_coeffs):
    is_esh = (frame_type == "ESH")
    if is_esh: X_subframes = frame_F_in.reshape(128, 8, order='F').T; num_subframes = 8
    else: X_subframes = frame_F_in.reshape(1, 1024); num_subframes = 1
    if tns_coeffs.ndim == 1: tns_coeffs = tns_coeffs.reshape(4, num_subframes)
    X_rec = np.zeros_like(X_subframes)
    for i in range(num_subframes):
        coeffs = tns_coeffs[:, i]
        X_rec[i, :] = lfilter([1.0], np.concatenate(([1.0], -coeffs)), X_subframes[i, :])
    if is_esh: return X_rec.T.flatten(order='F')
    return X_rec.flatten()

#LEVEL 3
#Psychoacoustic model helpers
def precompute_spreading_function(bands_info):
    num_bands = len(bands_info); bvals = bands_info[:, 4]; sf_mat = np.zeros((num_bands, num_bands))
    for i in range(num_bands):
        for j in range(num_bands):
            if i >= j: tmpx = 3.0 * (bvals[j] - bvals[i])
            else: tmpx = 1.5 * (bvals[j] - bvals[i])
            tmpz = 8 * min((tmpx - 0.5)**2 - 2*(tmpx - 0.5), 0)
            tmpy = 15.811389 + 7.5*(tmpx + 0.474) - 17.5 * np.sqrt(1.0 + (tmpx + 0.474)**2)
            if tmpy < -100:
                sf_mat[i, j] = 0.0
            else:
                sf_mat[i, j] = 10 ** ((tmpz + tmpy) / 10.0)
    return sf_mat

SPREADING_FUNCS_LONG = precompute_spreading_function(B219a)
SPREADING_FUNCS_SHORT = precompute_spreading_function(B219b)

def psycho_one_window(sample_data, prev1_data, prev2_data, bands_info, spreading_func):
    N = len(sample_data); n = np.arange(N); h_win = 0.5 - 0.5 * np.cos(np.pi * (n + 0.5) / N)
    sw = sample_data * h_win; fft_curr = fft(sw); r = np.abs(fft_curr)[:N//2]; f = np.angle(fft_curr)[:N//2]
    def get_rf(d):
        if d is None: return np.zeros(N//2), np.zeros(N//2)
        sw_p = d * h_win; fft_p = fft(sw_p); return np.abs(fft_p)[:N//2], np.angle(fft_p)[:N//2]
    r_1, f_1 = get_rf(prev1_data); r_2, f_2 = get_rf(prev2_data)
    r_pred = 2 * r_1 - r_2; f_pred = 2 * f_1 - f_2
    cw = np.sqrt((r*np.cos(f) - r_pred*np.cos(f_pred))**2 + (r*np.sin(f) - r_pred*np.sin(f_pred))**2) / (r + np.abs(r_pred) + EPS)
    num_bands = len(bands_info); e_b = np.zeros(num_bands); c_b = np.zeros(num_bands); max_bin = N // 2
    starts = bands_info[:, 1].astype(int); ends = bands_info[:, 2].astype(int) + 1
    if N == 2048 and starts[0] == 1: starts -= 1; ends -= 1
    starts = np.clip(starts, 0, max_bin); ends = np.clip(ends, 0, max_bin)
    for b in range(num_bands):
        if starts[b] >= ends[b]: continue
        sl = slice(starts[b], ends[b])
        e_b[b] = np.sum(r[sl]**2); c_b[b] = np.sum(cw[sl] * (r[sl]**2))
    ecb = np.dot(e_b, spreading_func); ct = np.dot(c_b, spreading_func)
    cb = np.clip(ct / (ecb + EPS), 1e-10, 0.99999); en = ecb / (np.sum(spreading_func, axis=0) + EPS)
    tb = np.clip(-0.299 - 0.43 * np.log(cb), 0, 1)
    SNR_loc = tb * 18.0 + (1 - tb) * 6.0; bc = 10 ** (-SNR_loc / 10.0); nb = en * bc
    qs_db = bands_info[:, 5]; q_thr_energy = EPS * (N/2) * (10 ** (qs_db / 10.0))
    npart = np.maximum(nb, q_thr_energy); SMR = e_b / (npart + EPS)
    return SMR

def psycho(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2):
    
    if frame_type == "ESH":
        SMR_matrix = np.zeros((len(B219b), 8))
        for k in range(8):
            s = 448 + k*128
            e = s + 256
            curr_sub = frame_T[s:e]

            
            if k == 0:
                prev_1 = frame_T_prev_1[448 + 7*128 : 448 + 7*128 + 256]
            else:
                prev_1 = frame_T[448 + (k-1)*128 : 448 + (k-1)*128 + 256]

            if k == 0:
                prev_2 = frame_T_prev_1[448 + 6*128 : 448 + 6*128 + 256]
            elif k == 1:
                prev_2 = frame_T_prev_1[448 + 7*128 : 448 + 7*128 + 256]
            else:
                prev_2 = frame_T[448 + (k-2)*128 : 448 + (k-2)*128 + 256]

            SMR_matrix[:, k] = psycho_one_window(curr_sub, prev_1, prev_2, B219b, SPREADING_FUNCS_SHORT)
        return SMR_matrix
        
    SMR = psycho_one_window(frame_T.flatten(), frame_T_prev_1.flatten(), frame_T_prev_2.flatten(), B219a, SPREADING_FUNCS_LONG)
    return SMR.reshape(-1, 1)
#Quantizer helpers
def quantize_coeffs(x, alpha):
    return (np.sign(x) * np.floor((np.abs(x) * 2**(-0.25*alpha))**0.75 + 0.4054)).astype(int)

def dequantize_coeffs(s, alpha):
    return np.sign(s) * (np.abs(s)**(4/3)) * 2**(0.25*alpha)

def aac_quantizer(frame_F, frame_type, SMR):
    is_esh = (frame_type == "ESH")
    if is_esh: bands_info = B219b; N = 128; num_sub = 8; frame_F_resh = frame_F
    else: bands_info = B219a; N = 1024; num_sub = 1; frame_F_resh = frame_F
    num_bands = len(bands_info)
    starts = bands_info[:, 1].astype(int); ends = bands_info[:, 2].astype(int) + 1
    if (not is_esh) and starts[0] == 1: starts -= 1; ends -= 1
    starts = np.clip(starts, 0, N); ends = np.clip(ends, 0, N)
    S_out = np.zeros((N, num_sub), dtype=int) if is_esh else np.zeros((N, 1), dtype=int)
    sfc_out = np.zeros((num_bands, num_sub), dtype=int); G_out = np.zeros(num_sub, dtype=int)

    for k in range(num_sub):
        X_k = frame_F_resh[:, k] if is_esh else frame_F_resh.flatten()
        SMR_k = SMR[:, k] if is_esh else SMR.flatten()
        T_b = np.zeros(num_bands)
        
        for b in range(num_bands):
            if starts[b] >= ends[b]: continue
            sl = slice(starts[b], ends[b])
            P_b = np.sum(X_k[sl]**2)
            if SMR_k[b] > 0: T_b[b] = P_b / SMR_k[b]
            else: T_b[b] = P_b 

        MQ = 8191; max_X = np.max(np.abs(X_k))
        if max_X < 1e-9: alpha_start = 0.0
        else: alpha_start = (16/3) * np.log2((max_X**0.75) / MQ)
        alpha_b = np.full(num_bands, int(np.floor(alpha_start)), dtype=int)

        for b in range(num_bands):
            if starts[b] >= ends[b]: continue
            sl = slice(starts[b], ends[b])
            X_band = X_k[sl]
            if X_band.size == 0: continue
            while True:
                s_band = quantize_coeffs(X_band, alpha_b[b])
                x_rec = dequantize_coeffs(s_band, alpha_b[b])
                pe = np.sum((X_band - x_rec)**2)
                if pe > T_b[b] * (1.0 + 1e-12): break
                alpha_b[b] += 1
                if np.all(s_band == 0): break 
            s_band = quantize_coeffs(X_band, alpha_b[b])
            x_rec = dequantize_coeffs(s_band, alpha_b[b])
            pe = np.sum((X_band - x_rec)**2)
            if pe > T_b[b] * (1.0 + 1e-12) and alpha_b[b] > int(np.floor(alpha_start)): alpha_b[b] -= 1

        
        for _ in range(10):
            changed = False
            for b in range(num_bands - 1):
                diff = alpha_b[b+1] - alpha_b[b]
                if diff > 60: 
                    alpha_b[b+1] = alpha_b[b] + 60
                    changed = True
                elif diff < -60: 
                    alpha_b[b] = alpha_b[b+1] + 60
                    changed = True
            if not changed: break

        G_out[k] = alpha_b[0]
        sfc_out[0, k] = alpha_b[0]
        for b in range(1, num_bands): sfc_out[b, k] = alpha_b[b] - alpha_b[b-1]
        for b in range(num_bands):
            if starts[b] >= ends[b]: continue
            sl = slice(starts[b], ends[b])
            if is_esh: S_out[sl, k] = quantize_coeffs(X_k[sl], alpha_b[b])
            else: S_out[sl, 0] = quantize_coeffs(X_k[sl], alpha_b[b])

    
    if is_esh: return S_out.flatten(order='F').reshape(-1, 1), sfc_out, G_out.reshape(1, -1)
    return S_out.flatten().reshape(-1, 1), sfc_out, np.array([G_out[0]])

def i_aac_quantizer(S, sfc, G, frame_type):
    is_esh = (frame_type == "ESH")
    
    S_flat = S.flatten() 
    
    if is_esh: bands_info = B219b; N = 128; num_sub = 8; S_resh = S_flat.reshape((N, num_sub), order='F')
    else: bands_info = B219a; N = 1024; num_sub = 1; S_resh = S_flat.reshape((N, 1))
    starts = bands_info[:, 1].astype(int); ends = bands_info[:, 2].astype(int) + 1
    if (not is_esh) and starts[0] == 1: starts -= 1; ends -= 1
    starts = np.clip(starts, 0, N); ends = np.clip(ends, 0, N)
    X_hat = np.zeros_like(S_resh, dtype=float)
    if np.ndim(G) == 0: G = [G]
    for k in range(num_sub):
        alphas = np.zeros(len(bands_info), dtype=int)
        alphas[0] = int(sfc[0, k])
        for b in range(1, len(bands_info)): alphas[b] = int(sfc[b, k]) + alphas[b-1]
        for b in range(len(bands_info)):
            if starts[b] >= ends[b]: continue
            sl = slice(starts[b], ends[b])
            X_hat[sl, k] = dequantize_coeffs(S_resh[sl, k], alphas[b])
    if is_esh: return X_hat.flatten(order='F')
    return X_hat.flatten()

def aac_coder_3(filename_in, filename_aac_coded):
    data, fs = sf.read(filename_in)
    if len(data.shape) == 1: data = np.column_stack((data, data))
    
    
    remainder = data.shape[0] % 1024
    if remainder != 0: data = np.vstack((data, np.zeros((1024 - remainder, 2))))
    zeros_pad = np.zeros((1024, 2)); data_padded = np.vstack((zeros_pad, data, zeros_pad))
    num_frames = (data_padded.shape[0] - 2048) // 1024 + 1
    prev_frames = {0: [np.zeros(2048), np.zeros(2048)], 1: [np.zeros(2048), np.zeros(2048)]}
    prev_frame_type = "OLS"
    aac_seq_3 = []
    
    global GLOBAL_PLOT_DATA
    GLOBAL_PLOT_DATA = [] 
    
    try: huff_lut = huff_utils.load_LUT("huffCodebooks.mat")
    except Exception as e: print(f"Error: {e}"); return []

    print(f"Encoding Level 3: {num_frames} frames...")

    for i in range(num_frames):
        start = i * 1024; frame_T = data_padded[start: start + 2048, :]
        if i < num_frames - 1: next_frame_T = data_padded[(i+1)*1024: (i+1)*1024 + 2048, :]
        else: next_frame_T = np.zeros_like(frame_T)
        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        win_type = "KBD"
        frame_F = filter_bank(frame_T, frame_type, win_type)
        if frame_type == "ESH":
            f_in_l = frame_F[:, 0].reshape(128, 8, order='F'); f_in_r = frame_F[:, 1].reshape(128, 8, order='F')
        else:
            f_in_l = frame_F[:, 0].reshape(1024, 1); f_in_r = frame_F[:, 1].reshape(1024, 1)
        
        frame_entry = {"frame_type": frame_type, "win_type": win_type, "chl": {}, "chr": {}}
        p_entry = {"chl": None, "chr": None} 

        for ch, f_in in enumerate([f_in_l, f_in_r]):
            ch_key = "chl" if ch == 0 else "chr"
            
            #Energy computation before TNS for the plot 
            E_in = np.sum(f_in**2, axis=0) + 1e-12 
            
            f_out, tns_coeffs = tns(f_in, frame_type)
            
            #Energy computation after TNS and gain
            E_out = np.sum(f_out**2, axis=0) + 1e-12 
            tns_gain_mean = np.mean(10 * np.log10(E_in / E_out)) 
            
            p1, p2 = prev_frames[ch]
            SMR = psycho(frame_T[:, ch], frame_type, p1, p2)
            prev_frames[ch] = [frame_T[:, ch], p1]
            
            is_esh = (frame_type == "ESH")
            bands_info = B219b if is_esh else B219a
            num_bands = len(bands_info)
            starts = bands_info[:, 1].astype(int); ends = bands_info[:, 2].astype(int) + 1
            if (not is_esh) and starts[0] == 1: starts -= 1; ends -= 1
            N_fft = 128 if is_esh else 1024
            starts = np.clip(starts, 0, N_fft); ends = np.clip(ends, 0, N_fft)
            
            T_save_temp = np.zeros((num_bands, 8 if is_esh else 1))
            P_save_temp = np.zeros((num_bands, 8 if is_esh else 1)) 
            
            SMR_use = SMR if is_esh else SMR.flatten()
            X_use = f_out
            if not is_esh: X_use = X_use.flatten()
            
            for ks in range(8 if is_esh else 1):
                cur_X = X_use[:, ks] if is_esh else X_use
                cur_SMR = SMR_use[:, ks] if is_esh else SMR_use
                for b in range(num_bands):
                    if starts[b]>=ends[b]: continue
                    sl = slice(starts[b], ends[b])
                    P_v = np.sum(cur_X[sl]**2)
                    P_save_temp[b, ks] = P_v
                    if cur_SMR[b] > 0: T_save_temp[b, ks] = P_v / cur_SMR[b]
                    else: T_save_temp[b, ks] = P_v
            
            if is_esh:
                frame_entry[ch_key]["T"] = np.mean(T_save_temp, axis=1).reshape(-1, 1)
                p_entry[ch_key] = np.mean(P_save_temp, axis=1).reshape(-1, 1)
            else:
                frame_entry[ch_key]["T"] = T_save_temp
                p_entry[ch_key] = P_save_temp
                
            #Store the TNS gain in the global variable for the plot
            p_entry[ch_key + "_tns_gain"] = tns_gain_mean

            S, sfc, G = aac_quantizer(f_out, frame_type, SMR)

            sfc_flat = sfc.flatten(order='F')
            force_cb_arg = 11 if np.max(np.abs(sfc_flat)) < 16 else None
            huff_sfc, _ = huff_utils.encode_huff(sfc_flat, huff_lut, force_codebook=force_cb_arg)

            S_flat = S.flatten(order='F') if frame_type == "ESH" else S.flatten()
            huff_stream, cb_stream = huff_utils.encode_huff(S_flat, huff_lut)

            frame_entry[ch_key]["tns_coeffs"] = tns_coeffs
            frame_entry[ch_key]["G"] = G
            frame_entry[ch_key]["sfc"] = huff_sfc
            frame_entry[ch_key]["stream"] = huff_stream
            frame_entry[ch_key]["codebook"] = cb_stream

        aac_seq_3.append(frame_entry)
        GLOBAL_PLOT_DATA.append(p_entry) 
        
        prev_frame_type = frame_type
        if i % 50 == 0: print(f"Encoded {i}/{num_frames}")

    scipy.io.savemat(filename_aac_coded, {"aac_seq_3": aac_seq_3}, oned_as='row')
    return aac_seq_3

def i_aac_coder_3(aac_seq_3, filename_out):
    huff_lut = huff_utils.load_LUT("huffCodebooks.mat")
    num_frames = len(aac_seq_3)
    output_buffer = np.zeros(((num_frames * 1024) + 2048, 2))
    print("Decoding Level 3...")
    for i, frame_data in enumerate(aac_seq_3):
        frame_type = frame_data["frame_type"]
        win_type = frame_data["win_type"]
        if isinstance(frame_type, np.ndarray): frame_type = frame_type[0]
        if isinstance(win_type, np.ndarray): win_type = win_type[0]
        frame_F_rec = np.zeros((1024, 2))
        for ch, ch_key in enumerate(["chl", "chr"]):
            ch_data = frame_data[ch_key]
            sfc_bits = ch_data["sfc"]
            if isinstance(sfc_bits, np.ndarray):
                if sfc_bits.size == 0: sfc_bits = ""
                else: sfc_bits = sfc_bits[0]
            
            sfc_dec = huff_utils.decode_huff(sfc_bits, huff_lut[11])

            cb_idx = int(ch_data["codebook"][0]) if isinstance(ch_data["codebook"], np.ndarray) else int(ch_data["codebook"])
            stream_bits = ch_data["stream"]
            if isinstance(stream_bits, np.ndarray): stream_bits = stream_bits[0]

            if cb_idx > 0:
                S_dec = np.array(huff_utils.decode_huff(stream_bits, huff_lut[cb_idx]), dtype=int)
            else:
                S_dec = np.zeros(1024, dtype=int)

            is_esh = (frame_type == "ESH")
            bands = len(B219b) if is_esh else len(B219a)
            cols = 8 if is_esh else 1
            need = bands * cols
            if len(sfc_dec) > need: sfc_dec = sfc_dec[:need]
            elif len(sfc_dec) < need: sfc_dec = np.pad(np.array(sfc_dec, dtype=int), (0, need - len(sfc_dec)))
            sfc_mat = np.array(sfc_dec, dtype=int).reshape((bands, cols), order='F')
            if len(S_dec) > 1024: S_dec = S_dec[:1024]
            elif len(S_dec) < 1024: S_dec = np.pad(S_dec, (0, 1024 - len(S_dec)))
            G = ch_data["G"].flatten() if isinstance(ch_data["G"], np.ndarray) else ch_data["G"]
            f_rec_quant = i_aac_quantizer(S_dec, sfc_mat, G, frame_type)
            tns_coeffs = ch_data["tns_coeffs"]
            if frame_type == "ESH": f_rec_in = f_rec_quant.reshape(128, 8, order='F')
            else: f_rec_in = f_rec_quant.reshape(1024, 1)
            f_rec_final = i_tns(f_rec_in, frame_type, tns_coeffs)
            if frame_type == "ESH": frame_F_rec[:, ch] = f_rec_final.flatten(order='F')
            else: frame_F_rec[:, ch] = f_rec_final.flatten()
        frame_T_rec = i_filter_bank(frame_F_rec, frame_type, win_type)
        output_buffer[i*1024: i*1024 + 2048, :] += frame_T_rec
        
    final_output = output_buffer[1024:-1024, :]
   
    
    if filename_out: sf.write(filename_out, final_output, 48000)
    return final_output

#Helper for PLOTTING the results
def plot_level3_diagnostics(aac_seq, orig_file, decoded_file):
    global GLOBAL_PLOT_DATA 
    
    orig, fs = sf.read(orig_file)
    dec, _ = sf.read(decoded_file)
    n = min(len(orig), len(dec))
    orig = orig[:n]
    dec = dec[:n]
    error = orig - dec
    t_all = np.arange(n)

    #plot 0:zoomed Waveform Comparison
    zoom_start = len(orig) // 2
    zoom_end = zoom_start + 1000 


    if zoom_end > len(orig):
        zoom_start = max(0, len(orig) - 1000)
        zoom_end = len(orig)

    t_zoom = t_all[zoom_start:zoom_end]

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.title(f'Zoomed Waveform Comparison - LEFT Channel (Samples {zoom_start} to {zoom_end})')
    plt.plot(t_zoom, orig[zoom_start:zoom_end, 0], label='Original', color='blue', linewidth=2, alpha=0.7)
    plt.plot(t_zoom, dec[zoom_start:zoom_end, 0], label='Decoded', color='orange', linewidth=2, linestyle='--', alpha=1.0)
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.subplot(2, 1, 2)
    plt.title(f'Zoomed Waveform Comparison - RIGHT Channel (Samples {zoom_start} to {zoom_end})')
    plt.plot(t_zoom, orig[zoom_start:zoom_end, 1], label='Original', color='green', linewidth=2, alpha=0.7)
    plt.plot(t_zoom, dec[zoom_start:zoom_end, 1], label='Decoded', color='magenta', linewidth=2, linestyle='--', alpha=1.0)
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    # PLOT 1: Waveform Comparison
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.title('Waveform Comparison - LEFT Channel')
    plt.plot(t_all, orig[:, 0], label='Original', color='blue', alpha=0.6)
    plt.plot(t_all, dec[:, 0], label='Decoded', color='orange', alpha=0.6, linestyle='--')
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.title('Waveform Comparison - RIGHT Channel')
    plt.plot(t_all, orig[:, 1], label='Original', color='green', alpha=0.6)
    plt.plot(t_all, dec[:, 1], label='Decoded', color='magenta', alpha=0.6, linestyle='--')
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # PLOT 2:QUANTIZATION ERROR
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.title('Overall Quantization Error (Noise) - LEFT Channel')
    plt.plot(t_all, error[:, 0], color='red', linewidth=0.5, alpha=0.8)
    plt.xlabel('Time [samples]')
    plt.ylabel('Error Amplitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.title('Overall Quantization Error (Noise) - RIGHT Channel')
    plt.plot(t_all, error[:, 1], color='purple', linewidth=0.5, alpha=0.8)
    plt.xlabel('Time [samples]')
    plt.ylabel('Error Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # PLOT 3: TNS PREDICTION GAIN PER FRAME
    tns_gain_L = [frame_data['chl_tns_gain'] for frame_data in GLOBAL_PLOT_DATA]
    tns_gain_R = [frame_data['chr_tns_gain'] for frame_data in GLOBAL_PLOT_DATA]
    frames_idx = np.arange(len(tns_gain_L))

    plt.figure(figsize=(14, 5))
    plt.title('TNS Prediction Gain per Frame (dB)')
    plt.plot(frames_idx, tns_gain_L, color='forestgreen', label='LEFT Channel Gain', linewidth=1.5)
    plt.plot(frames_idx, tns_gain_R, color='mediumorchid', label='RIGHT Channel Gain', linewidth=1.5, alpha=0.8)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Frame Index')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # PLOT 4: SIGNAL ENERGY vs MASKING THRESHOLD
    target_frame = -1
    for i, frame in enumerate(aac_seq):
        if frame['frame_type'] == 'OLS' and i > 10:
            if np.mean(frame['chl']['T']) > 1e-5:
                target_frame = i
                break

    if target_frame != -1:
        frame_data = aac_seq[target_frame]
        
        T_L = frame_data['chl']['T'].flatten()
        P_L = GLOBAL_PLOT_DATA[target_frame]['chl'].flatten()
        
        T_R = frame_data['chr']['T'].flatten()
        P_R = GLOBAL_PLOT_DATA[target_frame]['chr'].flatten()

        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.title(f'Energy P(b) vs Threshold T(b) - Frame {target_frame} (LEFT)')
        plt.plot(P_L, label='Signal Energy P(b)', color='blue', alpha=0.7, marker='s', markersize=3)
        plt.plot(T_L, label='Threshold T(b)', color='red', marker='o', markersize=4)
        plt.xlabel('Scale Factor Bands')
        plt.ylabel('Energy (Log Scale)')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title(f'Energy P(b) vs Threshold T(b) - Frame {target_frame} (RIGHT)')
        plt.plot(P_R, label='Signal Energy P(b)', color='green', alpha=0.7, marker='s', markersize=3)
        plt.plot(T_R, label='Threshold T(b)', color='magenta', marker='o', markersize=4)
        plt.xlabel('Scale Factor Bands')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    # PLOT 5: BITRATE & WINDOW TYPE (SSC) PER FRAME
    frame_types = [f['frame_type'] for f in aac_seq]
    type_map = {"OLS": 0, "LSS": 1, "ESH": 2, "LPS": 3}
    y_vals_ssc = [type_map[ft] for ft in frame_types]

    bits_per_frame = []
    for frame in aac_seq:
        bits = 0
        for ch in ['chl', 'chr']:
            if isinstance(frame[ch]['stream'], str):
                bits += len(frame[ch]['stream'])
            else:
                bits += len(frame[ch]['stream'][0])
            if isinstance(frame[ch]['sfc'], str):
                bits += len(frame[ch]['sfc'])
            else:
                bits += len(frame[ch]['sfc'][0])
        bits_per_frame.append(bits)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    
    ax1.set_title('Window Type (SSC) & Allocated Bits per Frame')
    ax1.step(range(len(y_vals_ssc)), y_vals_ssc, where='post', color='dodgerblue', linewidth=2, label='Window Type')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["OLS", "LSS", "ESH", "LPS"])
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Window Type (SSC)', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(bits_per_frame, color='darkorange', alpha=0.8, linewidth=1.5, label='Bits Used')
    ax2.set_ylabel('Bits Allocated', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    fig.tight_layout()
    plt.show()

def demo_aac_3(filename_in, filename_out, frame_aac_coded):
    aac_seq = aac_coder_3(filename_in, frame_aac_coded)
    x = i_aac_coder_3(aac_seq, filename_out)
    orig, fs = sf.read(filename_in)
    orig_len = len(orig)
    if len(x) > orig_len: x = x[:orig_len]
    min_len = min(len(orig), len(x))
    orig = orig[:min_len]; x = x[:min_len]
    noise = orig - x
    # SNR computation per channel
    print("\n--- SNR Results ---")
    for ch, ch_name in enumerate(["Left", "Right"]):
        signal_pow_ch = np.mean(orig[:, ch]**2) if len(orig.shape) > 1 else np.mean(orig**2)
        noise_pow_ch = np.mean(noise[:, ch]**2) if len(noise.shape) > 1 else np.mean(noise**2)

        snr_ch = safe_snr_db(signal_pow_ch, noise_pow_ch)
        print(f"--- {ch_name} Channel ---")
        print(f"  Signal Power: {signal_pow_ch:.6e}")
        print(f"  Noise Power : {noise_pow_ch:.6e}")
        print(f"  SNR         : {snr_ch:.2f} dB")

    # Total SNR 
    signal_pow = float(np.mean(orig**2))
    noise_pow = float(np.mean(noise**2))
    SNR = safe_snr_db(signal_pow, noise_pow)
    total_bits = 0
    for frame in aac_seq:
        for ch in ['chl', 'chr']:
            total_bits += bitlen(frame[ch].get('sfc', ""))
            total_bits += bitlen(frame[ch].get('stream', ""))
    duration = len(orig) / 48000.0
    bitrate = total_bits / max(duration, 1e-12)
    orig_bitrate = 48000 * 16 * 2
    compression = orig_bitrate / max(bitrate, 1e-12)
    print_snr("\nLevel 3 SNR", SNR)
    print(f"Bitrate: {bitrate/1000:.2f} kbps")
    print(f"Compression Ratio: {compression:.2f}:1")
    
    plot_level3_diagnostics(aac_seq, filename_in, filename_out)
    
    return SNR, bitrate, compression

if __name__ == "__main__":
    if os.path.exists('LicorDeCalandraca.wav'):
        demo_aac_3('LicorDeCalandraca.wav', 'output_lvl3.wav', 'encoded_lvl3.mat')
    else:
        print("File not found: LicorDeCalandraca.wav")