import time
from data_loader import DataLoader
from riesz_pyramid import RieszPyramid
from signal_processor import SignalProcessor
from reconstruction import AudioReconstructor


def main():
    # --- 参数设置 ---
    RAW_FILE = "output04.raw"
    DELTA_T_US = 500  # 500us -> 2000Hz 采样率
    SAMPLE_RATE = 1e6 / DELTA_T_US

    PYRAMID_LEVELS = 3  # Riesz 金字塔层数
    MAX_FRAMES_TO_PROCESS = 16000  # 例如 16000 帧 ~= 8 秒（2000Hz）

    # 推荐：根据目标弦的基频/泛音范围调整
    BAND_LOW_HZ = 50.0
    BAND_HIGH_HZ = 200.0

    TOP_K_PERCENT = 5.0  # 第一阶段：按 mean amplitude 取 top k% 像素
    REFINE_KEEP_PERCENT = 50.0  # 第二阶段：按“频带能量占比”再保留一部分

    print("==========================================")
    print("  事件相机声音复原项目 (Riesz + PCA)")
    print(f"  采样率: {SAMPLE_RATE:.1f} Hz")
    print(f"  处理限制: 前 {MAX_FRAMES_TO_PROCESS} 帧")
    print(f"  Band-pass: {BAND_LOW_HZ}-{BAND_HIGH_HZ} Hz")
    print("==========================================\n")

    t0 = time.time()

    # 1) 数据加载（signed frame + float32）
    loader = DataLoader()
    full_video = loader.load_and_generate_frames(
        RAW_FILE,
        delta_t_us=DELTA_T_US,
        max_frames=MAX_FRAMES_TO_PROCESS
    )

    if full_video.shape[0] == 0:
        print("错误：未加载到任何数据，请检查文件路径或 RAW 是否为空。")
        return

    # 2) Riesz 金字塔：输出 phase/amplitude（float32）
    pyramid_builder = RieszPyramid(levels=PYRAMID_LEVELS)
    phases, amplitudes = pyramid_builder.process_video_batch(full_video)

    # 3) 信号处理与特征矩阵构建（带通 + 标准化）
    processor = SignalProcessor(sample_rate=SAMPLE_RATE)
    feature_matrix = processor.extract_active_signals(
        phases,
        amplitudes,
        top_k_percent=TOP_K_PERCENT,
        band_low_hz=BAND_LOW_HZ,
        band_high_hz=BAND_HIGH_HZ,
        refine_by_band_energy=True,
        refine_keep_percent=REFINE_KEEP_PERCENT,
    )

    # 4) 音频重构（PCA）
    reconstructor = AudioReconstructor(sample_rate=SAMPLE_RATE)
    raw_audio = reconstructor.perform_pca_denoising(feature_matrix)
    final_audio = reconstructor.post_process_audio(
        raw_audio,
        band_low_hz=BAND_LOW_HZ,
        band_high_hz=BAND_HIGH_HZ
    )

    # 5) 保存与可视化
    reconstructor.save_to_wav(final_audio, "restored_sound.wav")
    reconstructor.visualize_result(final_audio, title="Event-based Audio Recovery", save_path="spectrogram.png")

    print("\n==========================================")
    print(f"  任务完成！总耗时: {time.time() - t0:.2f} 秒")
    print("==========================================")


if __name__ == "__main__":
    main()
