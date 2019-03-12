import numpy as np
import imageio as io
import matplotlib.pyplot as plt

img = io.imread('./0-Rechauffement/pouliot.jpg')
H1 = np.array([[0.9752, 0.0013, -100.3164],[-0.4886, 1.7240, 24.8480],[-0.0016, 0.0004, 1.0000]])
H2 = np.array([[0.1814, 0.7402, 34.3412],[1.0209, 0.1534, 60.3258],[0.0005, 0, 1.0000]])

def minMax(point):
    tmpX, tmpY = [], []
    for i in range(0,4):
        tmpX.append(point[i][0]/point[i][2])
        tmpY.append(point[i][1]/point[i][2])
    return np.amin(tmpX), np.amax(tmpX), np.amin(tmpY), np.amax(tmpY)

def appliqueTransformation(img, H):
    # Trouver le bon canevas
    pts = []
    pts = np.append(pts,((np.matmul(H,np.array([0,0,1]))).astype(float)))
    pts = np.append(pts,((np.matmul(H,np.array([347,0,1]))).astype(float)))
    pts = np.append(pts,((np.matmul(H,np.array([0,238,1]))).astype(float)))
    pts = np.append(pts,((np.matmul(H,np.array([347,238,1]))).astype(float)))
    pts = np.reshape(pts, (4,3))
    minX, maxX, minY, maxY = minMax(pts)
    # Calcul de l'image
    imgInt = np.zeros((int(maxY) - int(minY), int(maxX) - int(minX), 3),dtype=int)
    H_1 = np.linalg.inv(H)
    for i in range(int(minX), int(maxX)):
        for j in range(int(minY), int(maxY)):
            pts = np.matmul(H_1, np.array([i,j,1]))
            pts = (pts*(1/pts[2])).astype(int)
            if pts[0] < img.shape[1] and pts[1] < img.shape[0] and pts[0] >= 0 and pts[1] >= 0:
                imgInt[j - int(minY)][i - int(minX)] = img[pts[1]][pts[0]]
    return imgInt

# appliqueTransformation(img,H1)
imgTrans = appliqueTransformation(img,H1)
plt.imshow(imgTrans)
plt.show()

imgTrans = appliqueTransformation(img, H2)
plt.imshow(imgTrans)
plt.show()