#LVL 2
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import dct
from scipy.signal import lfilter
from scipy.signal.windows import kaiser
from scipy.linalg import toeplitz, solve
from numpy.polynomial import Polynomial
import os
import scipy.io

#Helper functions
def get_kbd_window(N, alpha=4.0):
    M = N // 2
    w = kaiser(M + 1, beta=np.pi * alpha)
    w_cumsum = np.cumsum(w)
    denom = w_cumsum[-1]
    win_left = np.sqrt(w_cumsum[:M] / denom)
    win_right = win_left[::-1]
    return np.concatenate((win_left, win_right))

def derive_window(win_type, N):
    if win_type == "KBD":
        return get_kbd_window(N, alpha=6.0 if N==2048 else 4.0)
    else:
        n = np.arange(N)
        return np.sin(np.pi / N * (n + 0.5))

def mdct4(x):
    N = x.shape[0] // 2
    n4 = N // 2
    a, b, c, d = x[0:n4], x[n4:2*n4], x[2*n4:3*n4], x[3*n4:4*n4]
    folded = np.concatenate((-c[::-1] - d, a - b[::-1]))
    return dct(folded, type=4, norm='ortho')

def imdct4(X):
    N = X.shape[0]
    n2 = N // 2
    y_folded = dct(X, type=4, norm='ortho')
    y0, y1 = y_folded[0:n2], y_folded[n2:N]
    return np.concatenate((y1, -y1[::-1], -y0[::-1], -y0))
#LVL 1 functions
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

#LVL 2 functions
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
        for k in range(1, N): Sw[k] = (Sw[k] + Sw[k-1]) / 2.0
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


def aac_coder_2(filename_in):
    data, fs = sf.read(filename_in)
    if len(data.shape) == 1: data = np.column_stack((data, data))

    n_samples = data.shape[0]
    remainder = n_samples % 1024
    if remainder != 0:
        data = np.vstack((data, np.zeros((1024 - remainder, 2))))
    zeros_pad = np.zeros((1024, 2))
    data_padded = np.vstack((zeros_pad, data, zeros_pad))

    num_frames = (data_padded.shape[0] - 2048) // 1024 + 1
    aac_seq_2 = []
    prev_frame_type = "OLS"
    print(f"Encoding Level 2: {num_frames} frames...")

    for i in range(num_frames):
        start = i * 1024
        frame_T = data_padded[start : start + 2048, :]
        if i < num_frames - 1: next_frame_T = data_padded[(i+1)*1024 : (i+1)*1024 + 2048, :]
        else: next_frame_T = np.zeros_like(frame_T)

        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        win_type = "KBD"
        frame_F = filter_bank(frame_T, frame_type, win_type)

        if frame_type == "ESH":
            f_in_left = frame_F[:, 0].reshape(128, 8, order='F')
            f_in_right = frame_F[:, 1].reshape(128, 8, order='F')
        else:
            f_in_left = frame_F[:, 0].reshape(1024, 1)
            f_in_right = frame_F[:, 1].reshape(1024, 1)

        f_out_left, tns_coeffs_left = tns(f_in_left, frame_type)
        f_out_right, tns_coeffs_right = tns(f_in_right, frame_type)

        frame_entry = {
            "frame_type": frame_type, "win_type": win_type,
            "chl": { "frame_F": f_out_left, "tns_coeffs": tns_coeffs_left },
            "chr": { "frame_F": f_out_right, "tns_coeffs": tns_coeffs_right }
        }
        aac_seq_2.append(frame_entry)
        prev_frame_type = frame_type

    return aac_seq_2

def i_aac_coder_2(aac_seq_2, filename_out):
    num_frames = len(aac_seq_2)
    output_buffer = np.zeros(((num_frames * 1024) + 2048, 2))
    print(f"Decoding Level 2...")

    for i, frame_data in enumerate(aac_seq_2):
        frame_type = frame_data["frame_type"]
        win_type = frame_data["win_type"]

        f_left = i_tns(frame_data["chl"]["frame_F"], frame_type, frame_data["chl"]["tns_coeffs"])
        f_right = i_tns(frame_data["chr"]["frame_F"], frame_type, frame_data["chr"]["tns_coeffs"])

        if frame_type == "ESH":
            f_left_flat = f_left.flatten(order='F'); f_right_flat = f_right.flatten(order='F')
        else:
            f_left_flat = f_left.flatten(); f_right_flat = f_right.flatten()

        frame_F_rec = np.column_stack((f_left_flat, f_right_flat))
        frame_T_rec = i_filter_bank(frame_F_rec, frame_type, win_type)
        output_buffer[i*1024 : i*1024 + 2048, :] += frame_T_rec

    x = output_buffer[1024:-1024, :]
    if filename_out: sf.write(filename_out, x, 48000)
    return x

def demo_aac_2(filename_in, filename_out):
    aac_seq = aac_coder_2(filename_in)
    x = i_aac_coder_2(aac_seq, filename_out)

    orig, fs = sf.read(filename_in)

    orig_len = len(orig)
    if len(x) > orig_len: x = x[:orig_len]
    min_len = min(len(orig), len(x))
    orig = orig[:min_len]
    x = x[:min_len]

    noise = orig - x

    #SNR COMPUTATION PER CHANNEL
    num_channels = orig.shape[1] if len(orig.shape) > 1 else 1

    for ch in range(num_channels):
        ch_name = "Left" if ch == 0 else "Right"
        signal_pow_ch = np.mean(orig[:, ch]**2) if num_channels > 1 else np.mean(orig**2)
        noise_pow_ch = np.mean(noise[:, ch]**2) if num_channels > 1 else np.mean(noise**2)

        snr_ch = np.inf if noise_pow_ch == 0 else 10 * np.log10(signal_pow_ch / noise_pow_ch)
        print(f"--- {ch_name} Channel ---")
        print(f"  Signal Power: {signal_pow_ch:.6e}")
        print(f"  Noise Power : {noise_pow_ch:.6e}")
        print(f"  SNR         : {snr_ch:.2f} dB")

    #Total SNR
    noise_pow = np.mean(noise**2)
    signal_pow = np.mean(orig**2)
    total_SNR = np.inf if noise_pow == 0 else 10 * np.log10(signal_pow / noise_pow)
    print(f"\nTOTAL Level 2 SNR = {total_SNR:.2f} dB\n")

    #Result plots
    print("Generating 3 Diagrams (Spectrum, Gain, Waveform) for BOTH channels...")

    #MDCT Spectrum Before vs After TNS
    types_to_find = ['OLS', 'LSS', 'LPS', 'ESH']
    found_indices = {}
    for i, entry in enumerate(aac_seq):
        ftype = entry['frame_type']
        if ftype in types_to_find and ftype not in found_indices:
            if np.sum(np.abs(entry['chl']['tns_coeffs'])) > 0.1:
                found_indices[ftype] = i
        if len(found_indices) == 4: break

    for ftype in types_to_find:
        if ftype not in found_indices:
            for i, entry in enumerate(aac_seq):
                if entry['frame_type'] == ftype: found_indices[ftype] = i; break

    # MDCT spectrum comparison before and after TNS for all frame types
    plt.figure(figsize=(18, 16))
    plt.suptitle("MDCT Spectrum Before vs After TNS (Left & Right Channels)", fontsize=16)

    for row_idx, ftype in enumerate(types_to_find):
        if ftype in found_indices:
            idx = found_indices[ftype]
            frame_data = aac_seq[idx]

            for ch_idx, ch_key in enumerate(['chl', 'chr']):
                ch_name = "Left" if ch_idx == 0 else "Right"
                coeffs = frame_data[ch_key]['tns_coeffs']

                rec = i_tns(frame_data[ch_key]['frame_F'], ftype, coeffs)

                if ftype == 'ESH':
                    X_after = frame_data[ch_key]['frame_F'].flatten(order='F')
                    X_before = rec.flatten(order='F')
                else:
                    X_after = frame_data[ch_key]['frame_F'].flatten()
                    X_before = rec.flatten()

                plt.subplot(4, 2, row_idx * 2 + ch_idx + 1)
                limit = 600
                plt.plot(np.abs(X_before[:limit]), 'b', label='Before TNS', alpha=0.6)
                plt.plot(np.abs(X_after[:limit]), 'r', label='After TNS', alpha=0.7)
                plt.title(f"{ftype} Frame (Idx {idx}) - {ch_name} Channel")
                plt.xlabel("Freq Bin"); plt.ylabel("Magnitude")
                plt.legend(loc='upper right'); plt.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show()

    #TNS Prediction Gain
    gains_L = []
    gains_R = []

    for entry in aac_seq:
        ftype = entry['frame_type']

        for ch_key, gain_list in zip(['chl', 'chr'], [gains_L, gains_R]):
            coeffs = entry[ch_key]['tns_coeffs']
            rec = i_tns(entry[ch_key]['frame_F'], ftype, coeffs)

            if ftype == 'ESH':
                X_after = entry[ch_key]['frame_F'].flatten(order='F')
                X_before = rec.flatten(order='F')
            else:
                X_after = entry[ch_key]['frame_F'].flatten()
                X_before = rec.flatten()

            en_before = np.sum(X_before**2)
            en_after = np.sum(X_after**2)
            if en_after < 1e-12: en_after = 1e-12
            gain_list.append(10 * np.log10(en_before / en_after))

    plt.figure(figsize=(14, 5))
    plt.plot(gains_L, color='forestgreen', label='Left Channel', linewidth=1.5)
    plt.plot(gains_R, color='mediumorchid', label='Right Channel', linewidth=1.5, alpha=0.8)
    plt.title("TNS Prediction Gain per Frame")
    plt.xlabel("Frame Index"); plt.ylabel("Gain (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3); plt.axhline(0, color='black')
    plt.tight_layout(); plt.show()

    # Waveform & Error
    time_axis = np.arange(len(orig)) / fs
    plt.figure(figsize=(16, 10))

    for ch_idx in range(num_channels):
        ch_name = "Left" if ch_idx == 0 else "Right"

        # Waveform Subplot
        plt.subplot(2, 2, ch_idx + 1)
        plt.plot(time_axis, orig[:, ch_idx] if num_channels > 1 else orig, label='Original', color='blue', alpha=0.7)
        plt.plot(time_axis, x[:, ch_idx] if num_channels > 1 else x, 'r--', alpha=0.6, label='Reconstructed (TNS)')
        plt.legend(loc='upper right'); plt.title(f"Waveform Comparison ({ch_name})"); plt.grid(alpha=0.3)
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
        plt.xlim(0, time_axis[-1])

        # Error Subplot
        plt.subplot(2, 2, ch_idx + 3)
        plt.plot(time_axis, noise[:, ch_idx] if num_channels > 1 else noise, 'g')
        plt.title(f"Error Signal ({ch_name})")
        plt.xlabel("Time (s)"); plt.grid(alpha=0.3); plt.ylabel("Amplitude")
        plt.xlim(0, time_axis[-1])

    plt.tight_layout(); plt.show()

    return total_SNR

if __name__ == "__main__":
    if os.path.exists('LicorDeCalandraca.wav'):
        demo_aac_2('LicorDeCalandraca.wav', 'output_lvl2.wav')
    else:
        print("File not found.")