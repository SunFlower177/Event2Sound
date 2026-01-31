import numpy as np
import cv2
import scipy.fftpack as fftpack
import time


class RieszPyramid:

    def __init__(self, levels=3):
        self.levels = levels
        self.riesz_masks = {}
        print(f"[RieszPyramid] 初始化完成，金字塔层数: {levels}")

    def build_laplacian_pyramid(self, frame):
        # 1. 构建高斯金字塔
        gaussian_pyr = [frame]
        curr_frame = frame
        for i in range(self.levels):
            curr_frame = cv2.pyrDown(curr_frame)
            gaussian_pyr.append(curr_frame)

        # 2. 构建拉普拉斯金字塔 (L_i = G_i - upsample(G_{i+1}))
        laplacian_pyr = []
        for i in range(self.levels):
            up = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=(gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
            lap = gaussian_pyr[i] - up
            laplacian_pyr.append(lap)

        # 最后一层直接用高斯金字塔的顶层
        laplacian_pyr.append(gaussian_pyr[-1])
        return laplacian_pyr

    def _get_riesz_masks(self, shape):
        # 缓存不同分辨率的 Riesz masks
        if shape in self.riesz_masks:
            return self.riesz_masks[shape]

        h, w = shape
        fy = fftpack.fftfreq(h).reshape(-1, 1)
        fx = fftpack.fftfreq(w).reshape(1, -1)
        radius = np.sqrt(fx * fx + fy * fy)

        radius[0, 0] = 1  # 避免除零
        mask1 = (-1j * fx) / radius
        mask2 = (-1j * fy) / radius

        self.riesz_masks[shape] = (mask1, mask2)
        return mask1, mask2

    def riesz_transform(self, image):
        img_fft = fftpack.fft2(image)
        mask1, mask2 = self._get_riesz_masks(image.shape)

        r1 = fftpack.ifft2(img_fft * mask1)
        r2 = fftpack.ifft2(img_fft * mask2)

        return r1, r2

    def get_local_amplitude_and_phase(self, image):
        r1, r2 = self.riesz_transform(image)

        amplitude = np.sqrt((image ** 2) + (np.abs(r1) ** 2) + (np.abs(r2) ** 2))
        phase = np.arctan2(np.sqrt((np.abs(r1) ** 2) + (np.abs(r2) ** 2)), image)

        return amplitude, phase

    def process_video_batch(self, video_stack):
        num_frames = video_stack.shape[0]
        print(f"[RieszPyramid] 开始处理 {num_frames} 帧数据...")

        start_time = time.time()

        pyr_phases_list = [[] for _ in range(self.levels)]
        pyr_amplitudes_list = [[] for _ in range(self.levels)]

        for t in range(num_frames):
            # 临时转 float32 进行计算
            frame = video_stack[t].astype(np.float32)

            pyr = self.build_laplacian_pyramid(frame)

            for i in range(self.levels):
                amp, ph = self.get_local_amplitude_and_phase(pyr[i])

                # 1. 清洗 NaN
                amp = np.nan_to_num(amp, nan=0.0)
                ph = np.nan_to_num(ph, nan=0.0)

                # 2. 清洗异常值（相位/幅值在后续 unwrap + 滤波 + PCA 中对数值精度很敏感）
                #    这里保持 float32，避免 float16 量化导致“相位台阶/抖动”，进而产生可听噪声。
                amp = np.nan_to_num(
                    amp.astype(np.float32),
                    nan=0.0,
                    posinf=np.finfo(np.float32).max,
                    neginf=0.0
                )
                ph = np.nan_to_num(
                    ph.astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                # （可选）如果发现极少数像素幅值异常巨大，可启用软裁剪：
                # max_amp = 1e6
                # np.clip(amp, -max_amp, max_amp, out=amp)

                pyr_phases_list[i].append(ph)
                pyr_amplitudes_list[i].append(amp)

            if (t + 1) % 100 == 0:
                elapsed = time.time() - start_time
                fps = (t + 1) / elapsed
                eta = (num_frames - t - 1) / fps
                print(f"  [RieszPyramid] 进度: {t + 1}/{num_frames} | 速度: {fps:.1f} fps")
                print(f"               ETA: {eta:.1f} 秒")

        print(f"[RieszPyramid] 处理完成！耗时: {time.time() - start_time:.2f}秒")
        print("[RieszPyramid] 正在堆叠结果数组...")

        results_ph = []
        results_amp = []

        for i in range(self.levels):
            ph_stack = np.stack(pyr_phases_list[i], axis=0)
            amp_stack = np.stack(pyr_amplitudes_list[i], axis=0)

            results_ph.append(ph_stack)
            results_amp.append(amp_stack)
            print(f"  - Level {i} 完成: {ph_stack.shape}")

        return results_ph, results_amp
