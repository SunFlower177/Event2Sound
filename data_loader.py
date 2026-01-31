import numpy as np
import os
import sys

# 导入 Metavision SDK
try:
    from metavision_core.event_io import EventsIterator
except ImportError:
    print("【错误】未检测到 metavision_core 库。")
    print("请确保已安装 Prophesee Metavision SDK。")
    print("如果您无法安装 SDK，请尝试使用 Prophesee 官方工具将 RAW 转换为 CSV 后再读取。")
    sys.exit(1)


class DataLoader:
    """从事件相机 RAW 读取事件流，并按固定时间窗生成伪帧序列。

    关键改动（提升声音复原稳定性）：
    - 使用 polarity 生成 signed frame：frame = (#pos events) - (#neg events)
      这样可以保留振动导致的正负交替结构，避免把闪烁/热像素噪声当成“能量”累加。
    - 伪帧输出 dtype 使用 float32（后续 Riesz 相位/unwrap/PCA 对精度敏感）。
    """

    def __init__(self, width=640, height=480):
        self.width = int(width)
        self.height = int(height)
        print(f"[DataLoader] 初始化完成: {self.width}x{self.height}")

    def load_and_generate_frames(self, file_path, delta_t_us=500, max_frames=16000):
        """读取 RAW 并生成形状为 (T, H, W) 的伪帧序列。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"RAW 文件不存在: {file_path}")

        print(f"[DataLoader] 初始化读取器...")
        mv_iterator = EventsIterator(file_path, delta_t=delta_t_us)

        video_frames = []
        print(f"[DataLoader] 开始加载数据... delta_t={delta_t_us}us, max_frames={max_frames}")

        for i, events in enumerate(mv_iterator):
            if i >= max_frames:
                break

            frame = self._events_to_image(events)
            video_frames.append(frame)

            if (i + 1) % 1000 == 0:
                print(f"  已处理 {i + 1} 帧...")

        if not video_frames:
            print("【警告】没有读取到任何事件帧。")
            return np.zeros((0, self.height, self.width), dtype=np.float32)

        print(f"[DataLoader] 正在堆叠数组 (Stacking)...")
        try:
            video_stack = np.stack(video_frames, axis=0).astype(np.float32)

            mem_size_mb = video_stack.nbytes / (1024 * 1024)
            print(f"[DataLoader] 数据加载完成：")
            print(f"  - 形状: {video_stack.shape}")
            print(f"  - 类型: {video_stack.dtype}")
            print(f"  - 内存占用: {mem_size_mb:.2f} MB")

            return video_stack

        except Exception as e:
            print(f"【内存错误】在堆叠数组时发生错误: {e}")
            print("建议减小 max_frames 参数。")
            raise

    def _events_to_image(self, events):
        """把一个时间窗内的事件流转换成一张“伪帧”。

        默认使用 signed frame（强烈建议）：
            frame = (#pos) - (#neg)

        如果事件里没有 polarity 字段，则退化为计数帧（效果会差一些）。
        """
        x = events['x'].astype(np.int32)
        y = events['y'].astype(np.int32)

        flat_indices = y * self.width + x

        # 兼容 polarity 字段：p 可能是 bool/0-1/-1-1
        if 'p' not in events.dtype.names:
            counts = np.bincount(flat_indices, minlength=self.width * self.height).astype(np.float32)
            return counts.reshape((self.height, self.width))

        p = events['p']
        pos_mask = p > 0

        pos_counts = np.bincount(flat_indices[pos_mask], minlength=self.width * self.height).astype(np.float32)
        neg_counts = np.bincount(flat_indices[~pos_mask], minlength=self.width * self.height).astype(np.float32)

        frame = (pos_counts - neg_counts).reshape((self.height, self.width))

        # 可选：轻微压缩动态范围，抑制热像素（保持单调，不破坏相位结构）
        # frame = np.sign(frame) * np.sqrt(np.abs(frame))

        return frame


# --- 单元测试 ---
if __name__ == "__main__":
    loader = DataLoader()

    # 使用您的文件名进行测试
    raw_file = "recording_2025-12-09_20-42-02.raw"

    # 测试模式：只读取前若干帧
    try:
        video = loader.load_and_generate_frames(raw_file, delta_t_us=500, max_frames=2000)

        if video.shape[0] > 0:
            print("\n测试成功！模块工作正常。")
            print(f"最大像素值: {np.max(video)}")
            print(f"平均像素值: {np.mean(video)}")
        else:
            print("\n测试失败：没有数据被加载。")

    except Exception as e:
        print(f"\n测试出错: {e}")
