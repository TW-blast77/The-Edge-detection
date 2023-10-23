import cv2
import numpy as np

def canny_edge_detection(image, low_threshold, high_threshold):
    # 1. 將圖像轉換為灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊以減少噪聲
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 計算梯度
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # 4. 計算梯度幅值和方向
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # 5. 非最大抑制
    suppressed = np.copy(gradient_magnitude)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = gradient_direction[i, j]
            if (angle >= -np.pi/8 and angle < np.pi/8) or (angle >= 7*np.pi/8) or (angle < -7*np.pi/8):
                if (gradient_magnitude[i, j] <= gradient_magnitude[i, j+1]) or (gradient_magnitude[i, j] <= gradient_magnitude[i, j-1]):
                    suppressed[i, j] = 0
            elif (angle >= np.pi/8 and angle < 3*np.pi/8) or (angle >= -7*np.pi/8 and angle < -5*np.pi/8):
                if (gradient_magnitude[i, j] <= gradient_magnitude[i-1, j+1]) or (gradient_magnitude[i, j] <= gradient_magnitude[i+1, j-1]):
                    suppressed[i, j] = 0
            elif (angle >= 3*np.pi/8 and angle < 5*np.pi/8) or (angle >= -5*np.pi/8 and angle < -3*np.pi/8):
                if (gradient_magnitude[i, j] <= gradient_magnitude[i+1, j]) or (gradient_magnitude[i, j] <= gradient_magnitude[i-1, j]):
                    suppressed[i, j] = 0
            else:
                if (gradient_magnitude[i, j] <= gradient_magnitude[i-1, j-1]) or (gradient_magnitude[i, j] <= gradient_magnitude[i+1, j+1]):
                    suppressed[i, j] = 0

    # 6. 雙閾值檢測
    edges = np.zeros_like(suppressed)
    edges[(suppressed >= low_threshold) & (suppressed <= high_threshold)] = 255

    return edges

# 調整低閾值和高閾值以適應你的圖像
low_threshold = input("enter your low_threshold")
high_threshold = 600
high_threshold = int(high_threshold)
low_threshold = int(low_threshold)
# 讀取圖像
image = cv2.imread('oko.jpg')

# 執行Canny邊緣檢測
edges = canny_edge_detection(image, low_threshold, high_threshold)
edges = cv2.resize(edges,(512,512),interpolation=cv2.INTER_AREA)
# 保存或顯示結果
cv2.imwrite('./Output_img/canny_edges_custom.jpg', edges)
cv2.imshow('Canny Edges (Custom)', edges)

# 等待用戶按下任意鍵，然後關閉視窗
cv2.waitKey(0)
print("enter q")

