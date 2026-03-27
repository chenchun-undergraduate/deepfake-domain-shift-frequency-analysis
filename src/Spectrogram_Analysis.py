import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import random


# =========================
# 配置
# =========================
DATASETS = {
    "real_original": "frames/original/train/real",
    "fake_original": "frames/original/train/fake",
    "real_crf28": "frames/crf28/train/real",
    "fake_crf28": "frames/crf28/train/fake",
    "real_crf35": "frames/crf35/train/real",
    "fake_crf35": "frames/crf35/train/fake"
}

SAMPLE_SIZE = 150
IMG_SIZE = 256  # 统一尺寸

# =========================
# 工具函数
# =========================
def plot_fft_difference(fft_results, key1, key2):
    diff = fft_results[key2] - fft_results[key1]

    plt.figure(figsize=(5,5))
    plt.imshow(diff, cmap='bwr')  # 红蓝图
    plt.title(f"FFT Difference ({key1} - {key2})")
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f"difference between {key1} and {key2}.png")
    plt.show()

def split_dct(dct_img, low_ratio=0.1):
    h, w = dct_img.shape
    lh, lw = int(h*low_ratio), int(w*low_ratio)

    low = np.zeros_like(dct_img)
    high = np.zeros_like(dct_img)

    low[:lh, :lw] = dct_img[:lh, :lw]
    high[lh:, lw:] = dct_img[lh:, lw:]

    return low, high

def dct_radial_energy(dct_img):
    h, w = dct_img.shape

    Y, X = np.ogrid[:h, :w]
    dist = X + Y 

    dist = dist.astype(int)

    counts = np.bincount(dist.ravel())
    sums = np.bincount(dist.ravel(), weights=(np.abs(dct_img) ** 2).ravel())

    radial = sums / (counts + 1e-8)

    return radial

def high_freq_ratio(dct_img, threshold=50):
    radial = dct_radial_energy(dct_img)

    total = np.sum(radial)
    high = np.sum(radial[threshold:])

    return high / total

def load_random_images(folder, n=100):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected = random.sample(files, min(n, len(files)))

    images = []
    for f in selected:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        images.append(img)
    return images

# =========================
# FFT
# =========================

def compute_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1e-8)
    return magnitude

# =========================
# DCT
# =========================

def compute_dct(img):
    return np.log(np.abs(dct(dct(img.T, norm='ortho').T, norm='ortho')) + 1e-8)

# =========================
# Radial Energy
# =========================

def radial_profile(data):
    center = np.array(data.shape) // 2
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / (nr + 1e-8)
    return radialprofile


def high_pass_filter(img, radius=30):
    h, w = img.shape
    center = (h//2, w//2)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X-center[1])**2 + (Y-center[0])**2)

    mask = dist > radius  # 只保留高频
    return img * mask


def plot_high_freq(fft_results):
    plt.figure(figsize=(15,5))

    for i, (k, v) in enumerate(fft_results.items()):
        high = high_pass_filter(v, radius=30)

        plt.subplot(1, len(fft_results), i+1)
        plt.imshow(np.log(1 + np.abs(high)), cmap='jet')
        plt.title(k)
        plt.savefig("high frequency FFT.png")
        plt.axis('off')

    plt.suptitle("High-Frequency Components Only")
    plt.show()


# =========================
# 主流程
# =========================

fft_results = {}
dct_results = {}
radial_results = {}

for name, path in DATASETS.items():
    print(f"Processing {name}...")

    imgs = load_random_images(path, SAMPLE_SIZE)

    fft_list = []
    dct_list = []

    for img in imgs:
        fft_list.append(compute_fft(img))
        dct_list.append(compute_dct(img))

    # 平均
    avg_fft = np.mean(fft_list, axis=0)
    avg_dct = np.mean(dct_list, axis=0)

    fft_results[name] = avg_fft
    dct_results[name] = avg_dct
    radial_results[name] = radial_profile(avg_fft)



# =========================
# 画 FFT 图
# =========================
plt.figure(figsize=(12,4))
for i, (k, v) in enumerate(fft_results.items()):
    plt.subplot(1, len(fft_results), i+1)
    plt.imshow(v, cmap='jet')
    plt.title(f"{k} FFT")
    plt.axis('off')
plt.tight_layout()
plt.savefig("fft_comparison.png")
plt.show()

# =========================
# 只显示FFT高频
# =========================
plot_high_freq(fft_results)

# =========================
# 画 DCT 图
# =========================
plt.figure(figsize=(12,4))
for i, (k, v) in enumerate(dct_results.items()):
    plt.subplot(1, len(fft_results), i+1)
    plt.imshow(v, cmap='jet')
    plt.title(f"{k} DCT")
    plt.axis('off')
plt.tight_layout()
plt.savefig("dct_comparison.png")
plt.show()

# =========================
# 画 DCT 高低频对比图
# =========================
plt.figure(figsize=(15,5))

for i, (k, v) in enumerate(dct_results.items()):
    low, high = split_dct(v)

    plt.subplot(2, len(dct_results), i+1)
    plt.imshow(np.log(1 + np.abs(v)), cmap='jet')
    plt.title(k)
    plt.axis('off')

    plt.subplot(2, len(dct_results), i+1+len(dct_results))
    plt.imshow(np.log(1 + np.abs(high)), cmap='jet')
    plt.title(k + " (High)")
    plt.axis('off')

plt.suptitle("DCT High-Frequency Comparison")
plt.savefig("dct_high_frequency.png")
plt.show()

# =========================
# 画 FFT Radial 曲线
# =========================
plt.figure()
for k, v in radial_results.items():
    plt.plot(v, label=k)

plt.legend()
plt.title("Radial Frequency Energy")
plt.xlabel("Frequency Radius")
plt.ylabel("Energy")
plt.savefig("FFT_radial_energy.png")
plt.show()

# =========================
# 画 DCT Radial 曲线
# =========================
plt.figure(figsize=(6,4))

for k, v in dct_results.items():
    radial = dct_radial_energy(v)
    plt.plot(radial[:100], label=k)

plt.legend()
plt.title("DCT Radial Energy")
plt.xlabel("Frequency Radius")
plt.ylabel("Energy")
plt.savefig("DCT_radial_energy.png")
plt.show()

# =========================
# FFT 压缩影响比较图 蓝色为频率增强，红色为频率减弱
# =========================
plot_fft_difference(fft_results, "real_original", "real_crf35")
plot_fft_difference(fft_results, "fake_original", "fake_crf35")
plot_fft_difference(fft_results, "real_original", "real_crf28")
plot_fft_difference(fft_results, "fake_original", "fake_crf28")

