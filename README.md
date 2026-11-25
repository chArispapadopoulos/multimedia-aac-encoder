# AAC Audio Encoder in Python

Simplified implementation of an MPEG-2/4 AAC audio encoder and decoder, developed for the **Multimedia Systems** course (academic year 2025–2026) at the School of Electrical and Computer Engineering, Aristotle University of Thessaloniki.

The encoder is built in three levels. Each level adds new stages on top of the previous one:

| Level | Stages | Type | Result on test signal |
|-------|--------|------|-----------------------|
| 1 | Sequence Segmentation Control (SSC), MDCT filterbank with KBD windows | Lossless | SNR ≈ 303.3 dB |
| 2 | Level 1 + Temporal Noise Shaping (TNS) | Lossless | SNR ≈ 303.3 dB |
| 3 | Level 2 + psychoacoustic model, non-uniform quantizer, Huffman coding | Lossy | SNR 8.67 dB, 214.57 kbps, compression ratio 7.16:1 |

The SNR of Levels 1 and 2 is limited only by floating-point precision, so the reconstruction is perfect. In Level 3 the quantization noise is shaped by the psychoacoustic model so that it stays below the masking threshold. The bitrate drops from 1536 kbps (48 kHz, 16-bit stereo) to about 215 kbps.

A full description of the implementation and a discussion of the results are given in the [report](report/Multimedia_Systems_Assignment_2025-2026_Report.pdf).

## Repository structure

```
.
├── level_1/
│   └── level_1.py              # SSC + filterbank (aac_coder_1, i_aac_coder_1, demo_aac_1)
├── level_2/
│   ├── level_2.py              # + TNS (aac_coder_2, i_aac_coder_2, demo_aac_2)
│   └── TableB219.mat           # scalefactor band tables (B219a, B219b)
├── level_3/
│   ├── level_3.py              # + psychoacoustic model, quantizer, Huffman (aac_coder_3, i_aac_coder_3, demo_aac_3)
│   ├── huff_utils.py           # Huffman coding utilities (provided with the assignment)
│   ├── huffCodebooks.mat       # Huffman codebooks (provided with the assignment)
│   └── TableB219.mat
├── report/
│   ├── Multimedia_Systems_Assignment_2025-2026_Report.pdf
│   └── latex/                  # LaTeX source and figures of the report
├── requirements.txt
└── README.md
```

Each level folder is self-contained, so it can be run on its own.

## Requirements

Python 3.9 or newer and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

The code expects a 48 kHz stereo WAV file named `LicorDeCalandraca.wav` inside the folder of the level being run. This is the test signal used for the assignment. It is not included in the repository, so place your own copy there (or any other 48 kHz stereo WAV file renamed accordingly).

```bash
cd level_3
python level_3.py
```

Each script runs the corresponding demo function, which encodes and decodes the file, prints the SNR per channel and in total, and shows the diagnostic plots. The output files are:

- Level 1: `output_lvl1.wav`
- Level 2: `output_lvl2.wav`
- Level 3: `output_lvl3.wav` and the encoded sequence `encoded_lvl3.mat`

The functions can also be called directly, for example:

```python
from level_3 import demo_aac_3
SNR, bitrate, compression = demo_aac_3("LicorDeCalandraca.wav", "output_lvl3.wav", "encoded_lvl3.mat")
```

The code was originally developed and tested in Google Colab. The scripts run the same way in a local Python environment. Level 3 takes about a minute on the test signal.

## Implementation overview

**Level 1.** The SSC stage high-pass filters the next frame and compares the energy of its eight 128-sample segments to detect attacks. A per-channel state machine (OLS → LSS → ESH → LPS) decides the window type, and the two channel decisions are merged into a common frame type. The filterbank applies KBD windows (α = 6 for long, α = 4 for short) and computes the MDCT through a DCT-IV. Synthesis uses the IMDCT and overlap-add.

**Level 2.** TNS normalizes the MDCT spectrum per band, computes 4th-order LPC coefficients from the autocorrelation (Toeplitz system), quantizes them with a step of 0.1 and checks the stability of the inverse filter. The spectrum is filtered with the FIR prediction filter in the encoder and restored with the inverse IIR filter in the decoder.

**Level 3.** The psychoacoustic model computes the SMR per scalefactor band from the predictability of the FFT spectrum, the spreading function and the absolute threshold of hearing. The quantizer searches the scalefactor of each band iteratively so that the quantization error stays below the masking threshold, with neighbouring scalefactors limited to a difference of 60. The quantized coefficients and the DPCM-coded scalefactors are then Huffman coded.

## Authors

- Georgios Ioannidis
- Panagiotis Charalampos Papadopoulos

## Acknowledgements

The files `huff_utils.py`, `huffCodebooks.mat` and `TableB219.mat` were provided by the course instructors as part of the assignment.
