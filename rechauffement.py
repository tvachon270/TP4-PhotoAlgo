import numpy as np
import imageio as io
import matplotlib.pyplot as plt

img = io.imread('./0-Rechauffement/pouliot.jpg')
H1 = np.array([[0.9752, 0.0013, -100.3164],[-0.4886, 1.7240, 24.8480],[-0.0016, 0.0004, 1.0000]])
H2 = np.array([[0.1814, 0.7402, 34.3412],[1.0209, 0.1534, 60.3258],[0.0005, 0, 1.0000]])

def appliqueTransformation(img, H):
    # Trouver le nouveau canevas
    pt1 = (np.matmul(H,np.array([0,0,1])))
    pt1 = (pt1/pt1[2]).astype(int)
    pt2 = (np.matmul(H,np.array([0,347,1])))
    pt2 = (pt2/pt2[2]).astype(int)
    pt3 = (np.matmul(H,np.array([238,0,1])))
    pt3 = (pt3/pt3[2]).astype(int)
    pt4 = (np.matmul(H,np.array([238,347,1])))
    pt4 = (pt4/pt4[2]).astype(int)
    minX = abs(np.amin(np.array([pt1[1], pt2[1], pt3[1], pt4[1]])))
    maxX = np.amax(np.array([pt1[1], pt2[1], pt3[1], pt4[1]]))
    minY = abs(np.amin(np.array([pt1[0], pt2[0], pt3[0], pt4[0]])))
    maxY = np.amax(np.array([pt1[0], pt2[0], pt3[0], pt4[0]]))
    imgInt = np.zeros((minY+maxY, minX+maxX, 3), dtype=int)
    for i in range(0,238):
        for j in range(0,347):
            pts = (np.matmul(H,np.array([i,j,1])))
            pts = (pts/pts[2]).astype(int)
            imgInt[pts[0]+minY][pts[1]+minX] = img[i][j]
    return imgInt



imgTrans = appliqueTransformation(img, H1)
plt.imshow(imgTrans)
plt.show()

# imgTrans = appliqueTransformation(img, H2)
# plt.imshow(imgTrans)
# plt.show()