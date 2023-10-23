import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('oko.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('F-16-image-example-H-0-1-255-512-512_Q320.jpg', cv2.IMREAD_GRAYSCALE)

# 确保图像加载成功
if img1 is None or img2 is None:
    raise Exception("Retry Open File")

#def sobel x的捲積核心
sobel_x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
#def sobel y的捲積核心
sobel_y = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

#def 拉普拉斯的捲積核心
laplacian = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

# def sobel function
def sobel_edge_detection(image):
    gradient_x = np.zeros_like(image, dtype=float)
    gradient_y = np.zeros_like(image, dtype=float)
    gradient_magnitude = np.zeros_like(image, dtype=float)
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            gx = np.sum(sobel_x * image[i-1:i+2, j-1:j+2])
            gy = np.sum(sobel_y * image[i-1:i+2, j-1:j+2])
            gradient_x[i, j] = gx
            gradient_y[i, j] = gy
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)
    
    return gradient_x, gradient_y, gradient_magnitude

edges_sobel_x1, edges_sobel_y1, edges_sobel1 = sobel_edge_detection(img1)
edges_sobel_x2, edges_sobel_y2, edges_sobel2 = sobel_edge_detection(img2)

# def Laplacian function
def laplacian_edge_detection(image):
    laplacian_output = np.zeros_like(image, dtype=float)
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            laplacian_value = np.sum(laplacian * image[i-1:i+2, j-1:j+2])
            laplacian_output[i, j] = laplacian_value
    
    return laplacian_output

edges_laplacian1 = laplacian_edge_detection(img1)
edges_laplacian2 = laplacian_edge_detection(img2)
cv2.imwrite('./Output_img/Edges(Sobel)1.jpg', edges_sobel1)
cv2.imwrite('./Output_img/Edges(Laplacian)1.jpg', edges_laplacian1)
cv2.imwrite('./Output_img/Edges(Sobel)2.jpg', edges_sobel2)
cv2.imwrite('./Output_img/Edges(Laplacian)2.jpg', edges_laplacian2)

#輸出影像
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(img1, cmap='gray'), plt.title('Original Image 1')
plt.subplot(232), plt.imshow(edges_sobel1, cmap='gray'), plt.title('Edges (Sobel) 1')
plt.subplot(233), plt.imshow(edges_laplacian1, cmap='gray'), plt.title('Edges (Laplacian) 1')
plt.subplot(234), plt.imshow(img2, cmap='gray'), plt.title('Original Image 2')
plt.subplot(235), plt.imshow(edges_sobel2, cmap='gray'), plt.title('Edges (Sobel) 2')
plt.subplot(236), plt.imshow(edges_laplacian2, cmap='gray'), plt.title('Edges (Laplacian) 2')
plt.show()
