import numpy as np
import scipy.signal as signal


class SignalProcessor:
    """信号处理器（内存安全版）

    核心修复：
    - 不再对 (T,H,W) 的 phase_video 做 np.unwrap（会爆内存）
    - 先按 amplitude 选出 mask，再抽取 (T,N) 后对 N 列 unwrap（内存大幅下降）
    """

    def __init__(self, sample_rate=2000.0):
        self.sample_rate = float(sample_rate)

    def unwrap_phase_2d(self, phase_matrix: np.ndarray) -> np.ndarray:
        """对 (T, N) 做 unwrap（内存安全）"""
        phase_matrix = phase_matrix.astype(np.float32, copy=False)
        return np.unwrap(phase_matrix, axis=0)

    def apply_band_pass_filter(self, x: np.ndarray, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
        low_hz = float(low_hz)
        high_hz = float(high_hz)

        nyq = 0.5 * self.sample_rate
        high_hz = min(high_hz, nyq * 0.95)

        if low_hz <= 0:
            raise ValueError(f"low_hz 必须 > 0, got {low_hz}")
        if not (low_hz < high_hz):
            raise ValueError(f"需要 low_hz < high_hz, got {low_hz}, {high_hz}")

        sos = signal.butter(N=order, Wn=[low_hz, high_hz], btype='bandpass', fs=self.sample_rate, output='sos')
        y = signal.sosfiltfilt(sos, x, axis=0)
        return y

    @staticmethod
    def robust_standardize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        med = np.nanmedian(X, axis=0, keepdims=True)
        mad = np.nanmedian(np.abs(X - med), axis=0, keepdims=True)
        scale = 1.4826 * mad
        scale = np.maximum(scale, eps)
        return (X - med) / scale

    def extract_active_signals(
        self,
        phases,
        amplitudes,
        top_k_percent: float = 5.0,
        band_low_hz: float = 60.0,
        band_high_hz: float = 800.0,
        refine_by_band_energy: bool = True,
        refine_keep_percent: float = 50.0,
        max_pixels_per_level: int  = 12000,
    ) -> np.ndarray:
        """返回 (T, N_active_pixels)"""

        print("[SignalProcessor] 开始提取有效振动信号...")
        collected = []

        for lvl, (ph_video, amp_video) in enumerate(zip(phases, amplitudes)):
            ph_video = np.nan_to_num(ph_video.astype(np.float32, copy=False))
            amp_video = np.nan_to_num(amp_video.astype(np.float32, copy=False))

            # 1) 先用 mean amplitude 做 mask（不 unwrap）
            mean_amp = np.mean(amp_video, axis=0)  # (H, W)
            if np.max(mean_amp) == 0:
                print(f"  - Level {lvl}: mean_amp 全 0，跳过。")
                continue

            threshold = np.nanpercentile(mean_amp, 100 - top_k_percent)
            if np.isnan(threshold):
                threshold = 0.0

            mask = mean_amp > threshold
            idx = np.flatnonzero(mask.ravel())
            count = int(idx.size)
            print(f"  - Level {lvl}: 阈值 {threshold:.4f}, 初筛像素: {count}")

            if count <= 0:
                continue

            # 2) 可选：限制最大像素数，避免 mask 过大导致后续 PCA 太慢
            if max_pixels_per_level is not None and count > max_pixels_per_level:
                # 取 mean_amp 最大的前 max_pixels_per_level 个点
                flat_amp = mean_amp.ravel()[idx]
                top_idx = np.argpartition(flat_amp, -max_pixels_per_level)[-max_pixels_per_level:]
                idx = idx[top_idx]
                count = int(idx.size)
                print(f"    Level {lvl}: 限制像素数 -> {count}")

            # 3) 抽取 (T, N) 相位序列（这里还没 unwrap）
            T = ph_video.shape[0]
            ph_flat = ph_video.reshape(T, -1)          # (T, H*W)
            active_phases = ph_flat[:, idx]            # (T, N)
            active_phases = np.nan_to_num(active_phases)

            # 4) 只对 (T,N) 做 unwrap（内存安全）
            active_phases = self.unwrap_phase_2d(active_phases)

            # 5) 带通滤波
            try:
                active_bp = self.apply_band_pass_filter(active_phases, band_low_hz, band_high_hz, order=4)
            except Exception as e:
                print(f"    [警告] Level {lvl}: 带通失败，退化为不滤波。原因: {e}")
                active_bp = active_phases

            active_bp = np.nan_to_num(active_bp)

            # 6) 二次筛选：频带能量占比
            if refine_by_band_energy and active_bp.shape[1] > 4:
                total_energy = np.mean(active_phases ** 2, axis=0) + 1e-12
                band_energy = np.mean(active_bp ** 2, axis=0)
                ratio = band_energy / total_energy

                keep_th = np.percentile(ratio, 100 - refine_keep_percent)
                keep = ratio >= keep_th

                before = active_bp.shape[1]
                active_bp = active_bp[:, keep]
                after = active_bp.shape[1]
                print(f"    Level {lvl}: 频带能量筛选保留 {after}/{before}")

                if after <= 0:
                    continue

            # 7) robust 标准化（PCA 前必须）
            active_bp = self.robust_standardize(active_bp)

            collected.append(active_bp)

        if not collected:
            print("【错误】未提取到任何有效信号！返回全零矩阵。")
            T = phases[0].shape[0] if len(phases) > 0 else 1
            return np.zeros((T, 1), dtype=np.float32)

        final_matrix = np.concatenate(collected, axis=1)
        final_matrix = np.nan_to_num(final_matrix).astype(np.float32, copy=False)

        print(f"[SignalProcessor] 特征提取完成。PCA 输入矩阵形状: {final_matrix.shape}")
        return final_matrix
