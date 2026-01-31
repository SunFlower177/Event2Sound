import numpy as np
from sklearn.decomposition import PCA
from scipy.io import wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
import os


class AudioReconstructor:

    def __init__(self, sample_rate=2000):
        self.sample_rate = sample_rate

    def perform_pca_denoising(self, signal_matrix, n_components=1):
        """PCA 降噪/特征提取：默认取第一主成分作为声音。

        注意：SignalProcessor 已对各列做了 robust 标准化，
        这里主要负责：
        - 清洗 NaN/Inf
        - 运行 PCA
        """
        signal_matrix = np.nan_to_num(signal_matrix).astype(np.float32)

        print(f"[Reconstruction] 正在进行 PCA 降噪... 输入矩阵: {signal_matrix.shape}")

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(signal_matrix)

        explained_variance = pca.explained_variance_ratio_
        print(f"  PCA 解释方差比: {explained_variance}")

        raw_audio = components[:, 0].astype(np.float32)
        return raw_audio

    def post_process_audio(self, raw_audio, band_low_hz=60.0, band_high_hz=800.0):
        """后处理：去趋势 + 带通 + 归一化。

        改动点：
        - 原先仅高通 20Hz，无法压制高频宽带事件噪声；
          这里改为带通（默认 60-800Hz，可按弦的基频/泛音调整）。
        - 使用 sosfiltfilt 做零相位滤波，减少“毛刺/颤动感”。
        """

        raw_audio = np.nan_to_num(raw_audio).astype(np.float32)

        # 1) 去除线性趋势（去超低频漂移）
        detrended = signal.detrend(raw_audio)

        # 2) 带通滤波
        nyq = 0.5 * self.sample_rate
        high = min(float(band_high_hz), nyq * 0.95)
        low = float(band_low_hz)
        if low <= 0:
            low = 1.0
        if low >= high:
            # 兜底：如果参数不合法，退化为高通
            sos = signal.butter(6, 20, 'hp', fs=self.sample_rate, output='sos')
            filtered = signal.sosfiltfilt(sos, detrended)
        else:
            sos = signal.butter(6, [low, high], 'bp', fs=self.sample_rate, output='sos')
            filtered = signal.sosfiltfilt(sos, detrended)

        # 3) 归一化（避免偶发尖峰撑爆幅度）
        peak = np.max(np.abs(filtered)) + 1e-12
        normalized = (filtered / peak).astype(np.float32)

        return normalized

    def save_to_wav(self, audio_data, filename="output.wav"):
        print(f"[Reconstruction] 正在保存音频到: {filename}")
        wavfile.write(filename, int(self.sample_rate), audio_data.astype(np.float32))

    def visualize_result(self, audio_data, title="Recovered Audio", save_path="result_plot.png"):
        print(f"[Reconstruction] 正在生成可视化图表...")

        audio_data = audio_data.astype(np.float32)
        time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))

        plt.figure(figsize=(12, 6))

        # 1. 波形图 (Waveform)
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, audio_data, color='blue', alpha=0.7)
        plt.title(f"{title} - Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)

        # 2. 声谱图 (Spectrogram)
        plt.subplot(2, 1, 2)
        Pxx, freqs, bins, im = plt.specgram(
            audio_data,
            NFFT=256,
            Fs=self.sample_rate,
            noverlap=128,
            cmap='inferno'
        )
        plt.title(f"{title} - Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label='dB')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"[Reconstruction] 图表已保存到: {save_path}")


# --- 模块四测试 ---
if __name__ == "__main__":
    np.random.seed(42)

    T = 1000
    clean_signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, T))
    noise_matrix = 0.5 * np.random.randn(T, 100)

    signal_matrix = clean_signal[:, np.newaxis] + noise_matrix

    reconstructor = AudioReconstructor(sample_rate=1000)

    recovered = reconstructor.perform_pca_denoising(signal_matrix)
    final_audio = reconstructor.post_process_audio(recovered)

    reconstructor.save_to_wav(final_audio, "test_output.wav")
    reconstructor.visualize_result(final_audio, save_path="test_vis.png")

    print("模块四测试完成。")
