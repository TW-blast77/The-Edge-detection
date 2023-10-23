# [影像邊緣偵測專案](#影像邊緣偵測專案)

这個GitHub專案包含了三種不同的影像邊緣偵測方法的示例程式碼和輸出結果。您可以在這裡找到如何使用拉普拉斯和Sobel運算子以及Canny邊緣偵測來提取影像的邊緣。

## 邊緣偵測方法

### 1. 拉普拉斯和Sobel邊緣偵測

- 使用3x3和5x5的拉普拉斯核心進行邊緣偵測。
- 使用Sobel運算子進行邊緣偵測。
- 程式碼範例在 `edge_detection.py` 中。

### 2. Canny邊緣偵測

- 使用Canny邊緣偵測演算法來檢測影像邊緣。
- 程式碼範例在 `canny_edge_detection.py` 中。

## 示範圖片

此專案包含以下示範圖片，您可以使用它們來測試邊緣偵測方法：

- `F-16-image-example-H-0-1-255-512-512_Q320.jpg`
- `F16.jpg`
- `oko.jpg`

## 輸出圖片

邊緣偵測的結果將保存在名為 `Output_img` 的資料夾中。每種邊緣偵測方法的輸出圖片將以相應的檔名保存在此資料夾中。

## 如何使用

1. 安裝所需的相依庫（例如，OpenCV、NumPy、Matplotlib）。
2. 執行 `edge_detection.py` 以執行拉普拉斯和Sobel邊緣偵測。
3. 執行 `canny_edge_detection.py` 以執行Canny邊緣偵測。
4. 輸出的圖片將保存在 `Output_img` 資料夾中。

## 示範

以下是示範圖片和對應的邊緣偵測結果：

![原始影像 1](oko.jpg)

**邊緣（Sobel）1**
![邊緣（Sobel）1](Output_img/Edges(Sobel)1.jpg)

**邊緣（Laplacian）1**
![邊緣（Laplacian）1](Output_img/Edges(Laplacian)1.jpg)

![原始影像 2](F-16-image-example-H-0-1-255-512-512_Q320.jpg)

**邊緣（Sobel）2**
![邊緣（Sobel）2](Output_img/Edges(Sobel)2.jpg)

**邊緣（Laplacian）2**
![邊緣（Laplacian）2](Output_img/Edges(Laplacian)2.jpg)

## 作者

[TW-Blast77]
