import numpy as np
import imageio as io
import scipy.misc as misc
import matplotlib.pyplot as plt


img = io.imread('./0-Rechauffement/pouliot.jpg')
H1 = np.array([[0.9752, 0.0013, -100.3164],[-0.4886, 1.7240, 24.8480],[-0.0016, 0.0004, 1.0000]])
H2 = np.array([[0.1814, 0.7402, 34.3412],[1.0209, 0.1534, 60.3258],[0.0005, 0, 1.0000]])

def pval(point, minX, minY, maxX, maxY):
    # if minX > point[0]:
    #     minX = point[0]
    # if minY > point[1]:
    #     minY = point[1]
    # if maxX < point[0]:
    #     maxX = point[0]
    # if maxY < point[1]:
    #     maxY = point[1]
    # if minX > point[1]:
    #     minX = point[1]
    # if minY > point[0]:
    #     minY = point[0]
    # if maxX < point[1]:
    #     maxX = point[1]
    # if maxY < point[0]:
    #     maxY = point[0]
    if minX > point[1]/point[2]:
        minX = point[1]/point[2]
    if minY > point[0]/point[2]:
        minY = point[0]/point[2]
    if maxX < point[1]/point[2]:
        maxX = point[1]/point[2]
    if maxY < point[0]/point[2]:
        maxY = point[0]/point[2]
    return minX, minY, maxX, maxY

def appliqueTransformation(img, H):
    # Trouver le bon canevas
    minX, minY, maxX, maxY = 0, 0, 0, 0
    pts = ((np.matmul(H,np.array([0,0,1]))).astype(float))
    print(pts)
    minX, minY, maxX, maxY = pval(pts, minX, minY, maxX, maxY)
    pts = ((np.matmul(H,np.array([0,347,1]))).astype(float))
    print(pts)
    minX, minY, maxX, maxY = pval(pts, minX, minY, maxX, maxY)
    pts = ((np.matmul(H,np.array([238,0,1]))).astype(float))
    print(pts)
    minX, minY, maxX, maxY = pval(pts, minX, minY, maxX, maxY)
    pts = ((np.matmul(H,np.array([238,347,1]))).astype(float))
    print(pts)
    minX, minY, maxX, maxY = pval(pts, minX, minY, maxX, maxY)
    print(minX)
    print(minY)
    print(maxX)
    print(maxY)
    # Calcul de l'image
    imgInt = np.zeros((int(abs(minY) + maxY), int(abs(minX) + maxX), 3),dtype=int)
    for i in range(0,238):
        for j in range(0,347):
            pts = (np.matmul(H,np.array([i,j,1])))
            ptsTrans = (pts/pts[2]).astype(int)
            imgInt[ptsTrans[0] + int(abs(minY))][ptsTrans[1] + int(abs(minX))] = img[i][j]
    return imgInt

imgTrans = appliqueTransformation(img, H1)
plt.imshow(imgTrans)
plt.show()

# imgTrans = appliqueTransformation(img, H2)
# plt.imshow(imgTrans)
# plt.show()
