#LVL 1
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import dct
from scipy.signal import lfilter
from scipy.signal.windows import kaiser
import os

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
        return get_kbd_window(N, alpha=6.0 if N == 2048 else 4.0)
    n = np.arange(N)
    return np.sin(np.pi / N * (n + 0.5))

def mdct4(x):
    N = x.shape[0] // 2
    n4 = N // 2
    a = x[0:n4]
    b = x[n4:2*n4]
    c = x[2*n4:3*n4]
    d = x[3*n4:4*n4]
    folded = np.concatenate((-c[::-1] - d, a - b[::-1]))
    return dct(folded, type=4, norm='ortho')

def imdct4(X):
    N = X.shape[0]
    n2 = N // 2
    y_folded = dct(X, type=4, norm='ortho')
    y0 = y_folded[0:n2]
    y1 = y_folded[n2:N]
    y_extended = np.concatenate((y1, -y1[::-1], -y0[::-1], -y0))
    return y_extended

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

def aac_coder_1(filename_in):
    data, samplerate = sf.read(filename_in)
    if len(data.shape) == 1:
        data = np.column_stack((data, data))

    n_samples = data.shape[0]
    remainder = n_samples % 1024
    if remainder != 0:
        pad_len = 1024 - remainder
        zeros_fill = np.zeros((pad_len, data.shape[1]))
        data = np.vstack((data, zeros_fill))

    zeros_pad = np.zeros((1024, 2))
    data_padded = np.vstack((zeros_pad, data, zeros_pad))

    num_frames = (data_padded.shape[0] - 2048) // 1024 + 1

    aac_seq_1 = []
    prev_frame_type = "OLS"

    for i in range(num_frames):
        start = i * 1024
        end = start + 2048
        frame_T = data_padded[start:end, :]

        if i < num_frames - 1:
            next_start = (i + 1) * 1024
            next_frame_T = data_padded[next_start : next_start + 2048, :]
        else:
            next_frame_T = np.zeros_like(frame_T)

        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        win_type = "KBD"
        frame_F = filter_bank(frame_T, frame_type, win_type)

        frame_entry = {
            "frame_type": frame_type,
            "win_type": win_type,
            "chl": {"frame_F": frame_F[:, 0].reshape(-1, 1)},
            "chr": {"frame_F": frame_F[:, 1].reshape(-1, 1)}
        }

        if frame_type == "ESH":
             frame_entry["chl"]["frame_F"] = frame_F[:, 0].reshape(128, 8, order='F')
             frame_entry["chr"]["frame_F"] = frame_F[:, 1].reshape(128, 8, order='F')

        aac_seq_1.append(frame_entry)
        prev_frame_type = frame_type

    return aac_seq_1

def i_aac_coder_1(aac_seq_1, filename_out):
    num_frames = len(aac_seq_1)
    output_buffer_len = (num_frames * 1024) + 1024
    output_buffer = np.zeros((output_buffer_len + 2048, 2))

    for i, frame_data in enumerate(aac_seq_1):
        frame_type = frame_data["frame_type"]
        win_type = frame_data["win_type"]
        f_left = frame_data["chl"]["frame_F"]
        f_right = frame_data["chr"]["frame_F"]

        if frame_type == "ESH" and f_left.shape == (128, 8):
             f_left = f_left.flatten(order='F')
             f_right = f_right.flatten(order='F')

        frame_F = np.column_stack((f_left, f_right))
        frame_T_rec = i_filter_bank(frame_F, frame_type, win_type)

        start = i * 1024
        end = start + 2048
        output_buffer[start:end, :] += frame_T_rec

    final_output = output_buffer[1024 : -1024, :]
    if filename_out:
        sf.write(filename_out, final_output, 48000)
    return final_output

def demo_aac_1(filename_in, filename_out):
   
    print(f"Encoding {filename_in}...")
    aac_seq = aac_coder_1(filename_in)

    print(f"Decoding to {filename_out}...")
    x = i_aac_coder_1(aac_seq, filename_out)

    orig, fs = sf.read(filename_in)

    orig_len = len(orig)
    if len(x) > orig_len: x = x[:orig_len]
    min_len = min(len(orig), len(x))
    orig = orig[:min_len]
    x = x[:min_len]

    noise = orig - x

    #SNR computation per channel (0: Left, 1: Right)
    for ch, ch_name in enumerate(["Left", "Right"]):
        signal_pow_ch = np.mean(orig[:, ch]**2) if len(orig.shape) > 1 else np.mean(orig**2)
        noise_pow_ch = np.mean(noise[:, ch]**2) if len(noise.shape) > 1 else np.mean(noise**2)

        snr_ch = np.inf if noise_pow_ch == 0 else 10 * np.log10(signal_pow_ch / noise_pow_ch)
        print(f"--- {ch_name} Channel ---")
        print(f"  Signal Power: {signal_pow_ch:.6e}")
        print(f"  Noise Power : {noise_pow_ch:.6e}")
        print(f"  SNR         : {snr_ch:.2f} dB")

    #Total SNR
    noise_pow = np.mean(noise**2)
    signal_pow = np.mean(orig**2)
    total_SNR = np.inf if noise_pow == 0 else 10 * np.log10(signal_pow / noise_pow)
    print(f"\nTOTAL SNR = {total_SNR:.2f} dB\n")

    #Result plots
    #Figure 1
    time_axis = np.arange(len(orig)) / fs
    plt.figure(figsize=(16, 16))

    num_channels = orig.shape[1] if len(orig.shape) > 1 else 1

    for ch in range(num_channels):
        ch_name = "Left" if ch == 0 else "Right"
        col_idx = ch + 1

        #Plot 1:Waveform Comparison
        plt.subplot(4, 2, col_idx)
        plt.plot(time_axis, orig[:, ch] if num_channels > 1 else orig, 'b', alpha=0.7, label='Original')
        plt.plot(time_axis, x[:, ch] if num_channels > 1 else x, 'r--', alpha=0.6, label='Reconstructed')
        plt.title(f'Waveform ({ch_name} Channel)')
        plt.legend(loc='upper right')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, time_axis[-1])

        #Plot 2:Error
        plt.subplot(4, 2, col_idx + 2)
        plt.plot(time_axis, noise[:, ch] if num_channels > 1 else noise, 'g')
        plt.title(f'Error ({ch_name} Channel)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, time_axis[-1])

        #Plot 3:Spectrogram for the first 20,000 samples
        plt.subplot(4, 2, col_idx + 4)
        plt.specgram(orig[:20000, ch] if num_channels > 1 else orig[:20000], Fs=fs, NFFT=1024, noverlap=512)
        plt.title(f'Spectrogram for 20k first samples({ch_name} Channel)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        #Plot 4:Zoomed Waveform Comparison
        plt.subplot(4, 2, col_idx + 6)
        zoom_len = 80
        if len(orig) > zoom_len:
            t_zoom = time_axis[-zoom_len:]
            orig_zoom = orig[-zoom_len:, ch] if num_channels > 1 else orig[-zoom_len:]
            x_zoom = x[-zoom_len:, ch] if num_channels > 1 else x[-zoom_len:]
            plt.plot(t_zoom, orig_zoom, 'b.-', label='Original')
            plt.plot(t_zoom, x_zoom, 'r.--', label='Reconstructed')
            plt.title(f'Zoomed Waveform Comparison ({ch_name} Channel)')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)

    plt.tight_layout()
    plt.show()

    #Figure 2: SSC DECISIONS
    frame_types = [frame['frame_type'] for frame in aac_seq]

    type_map = {"OLS": 0, "LSS": 1, "ESH": 2, "LPS": 3}
    y_vals = [type_map[t] for t in frame_types]
    x_frames = np.arange(len(frame_types))

    plt.figure(figsize=(15, 5))
    plt.step(x_frames, y_vals, where='post', color='purple', linewidth=2)
    plt.yticks([0, 1, 2, 3], ["OLS", "LSS", "ESH", "LPS"])
    plt.xlabel('FRAMES')
    plt.ylabel('FRAME TYPE')
    plt.title('Sequence Segmentation Control (SSC) Decisions')
    plt.grid(True, alpha=0.3)
    plt.show()

    def draw_dim_line(start, end, y_pos, text, color='black'):
        plt.annotate('', xy=(start, y_pos), xytext=(end, y_pos),
                     arrowprops=dict(arrowstyle='<->', color=color))
        plt.text((start + end)/2, y_pos - 0.02, text, color=color,
                 ha='center', va='top', fontsize=10, fontweight='bold')

    if len(aac_seq) > 26:
        frames_to_plot = [24, 25, 26]
        colors = ['blue', 'orange', 'red']
        styles = ['-', '-', '--']

        plt.figure(figsize=(18, 7))
        win_long = derive_window("KBD", 2048)
        win_short = derive_window("KBD", 256)

        for i, frame_idx in enumerate(frames_to_plot):
            frame_data = aac_seq[frame_idx]
            ftype = frame_data['frame_type']
            start_sample = frame_idx * 1024
            color = colors[i]
            style = styles[i]
            label_txt = f"Frame {frame_idx}: {ftype}"

            x_axis = np.arange(start_sample, start_sample + 2048)

            if ftype == "OLS":
                plt.plot(x_axis, win_long, color=color, linestyle=style, linewidth=2, label=label_txt)

            elif ftype == "LSS":
                w = np.zeros(2048)
                w[:1024] = win_long[:1024]
                w[1024:1472] = 1.0
                w[1472:1600] = win_short[128:]
                w[1600:] = 0.0
                plt.plot(x_axis, w, color=color, linestyle=style, linewidth=2, label=label_txt)
                draw_dim_line(start_sample + 1024, start_sample + 1472, -0.15, "448", color=color)
                draw_dim_line(start_sample + 1600, start_sample + 2048, -0.15, "448", color=color)

            elif ftype == "LPS":
                w = np.zeros(2048)
                w[:448] = 0.0
                w[448:576] = win_short[:128]
                w[576:1024] = 1.0
                w[1024:] = win_long[1024:]
                plt.plot(x_axis, w, color=color, linestyle=style, linewidth=2, label=label_txt)

            elif ftype == "ESH":
                start_internal = 448
                for k in range(8):
                    idx = start_sample + start_internal + k * 128
                    sub_x = np.arange(idx, idx + 256)
                    lbl = label_txt if k == 0 else None
                    plt.plot(sub_x, win_short, color=color, linestyle=style, label=lbl)
                draw_dim_line(start_sample + 448, start_sample + 1600, -0.15, "1152", color=color)

            plt.vlines(start_sample, -0.05, 0.05, colors='black', linewidth=1.5)
            plt.text(start_sample, -0.08, f"Fr {frame_idx}\n{start_sample}", color='black', fontsize=9, ha='center', va='top')

        plt.title('Detailed Frame Overlap (Frames 24 to 26)')
        plt.xlabel('SAMPLES')
        plt.ylabel('WINDOW AMPLITUDE')
        plt.ylim(-0.3, 1.1)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.show()

    return total_SNR

if __name__ == "__main__":
  if os.path.exists('LicorDeCalandraca.wav'):
    SNR = demo_aac_1('LicorDeCalandraca.wav', 'output_lvl1.wav')
  else:
    print("ΠΡΟΣΟΧΗ: Το αρχείο 'LicorDeCalandraca.wav' δεν βρέθηκε")