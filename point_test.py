import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


def stack_frames(frames):
    """疊加15個frame"""
    # 堆叠frames，新的堆叠维度是第三维
    return np.dstack(frames)


def calculate_entropy(stacked_frames):
    """计算每个pixel的entropy"""
    # 计算每个像素在堆叠的frames中的熵
    entropy_map = np.apply_along_axis(lambda x: entropy(x, base=2), axis=2, arr=stacked_frames)
    return entropy_map


def plot_heatmap(entropy_map):
    """画出热区图"""
    # 使用matplotlib绘制热图
    plt.imshow(entropy_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


def get_frames(num_frames=15, width=25, height=25):
    """获取frames"""
    # 假设这里是获取实际frame的代码，现在我们随机生成假数据
    return [np.random.randint(0, 256, size=(width, height), dtype=np.uint8) for _ in range(num_frames)]


# 主流程
frames = get_frames()
stacked_frames = stack_frames(frames)
entropy_map = calculate_entropy(stacked_frames)
plot_heatmap(entropy_map)
