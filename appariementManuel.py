import numpy as np
import imageio as io
import rechauffement as meth
import matplotlib.pyplot as plt

# Création d'une matrice 4x3 pour les points de bases
def matriceNx3(ptsBase):
    A = []
    n = ptsBase.shape[0]
    for i in range(0,n):
        pts = np.zeros((1,3), dtype=int)
        # point i
        np.put(pts, [0,1], ptsBase[i])
        np.put(pts, [2], 1)
        A .append(pts)
    A = np.reshape(A, (n,3))
    return A

# Création d'une matrice pour trouver la transformation projective
# Places les points entre deux images dans une matrice 8x9
def matriceResolve(ptsBase, ptsProj):
    A = []
    n = ptsBase.shape[0]
    for i in range(0,n):
        pts = np.zeros((2,9), dtype=int)
        # points i
        np.put(pts, [0,1,2], -1*(ptsBase[i]))
        np.put(pts, [6,7,8], ptsBase[i]*ptsProj[i][0])
        np.put(pts, [12,13,14], -1*(ptsBase[i]))
        np.put(pts, [15,16,17], ptsBase[i]*ptsProj[i][1])
        A.append(pts)
    A = np.reshape(A, (2*n,9))
    return A

# Calculer l'homographie
def calculHomographie(ptsBase, ptsProj):
    # Création de la matrice A
    ptsBase = matriceNx3(ptsBase)
    A = matriceResolve(ptsBase, ptsProj)
    # Décomposition en valeur singulière
    U,S,V = np.linalg.svd(A)
    # Extraction de la transformée H
    h = V[np.argmin(S)]
    h = np.reshape(h, (3,3))
    return h


if __name__ == "__main__":
    np.set_printoptions(precision=3,suppress=True)
    # Extraction des images
    img1 = io.imread('./1-PartieManuelle/Serie1/IMG_2415.jpg')
    img2 = io.imread('./1-PartieManuelle/Serie1/IMG_2416.jpg')
    img3 = io.imread('./1-PartieManuelle/Serie1/IMG_2417.jpg')

    # Extraction des points des images
    ptsBase1 = np.loadtxt("./1-PartieManuelle/Serie1/pts_serie1/pts2_12.txt", delimiter=",")
    tmp = np.copy(ptsBase1)
    ptsBase1[:, 0] = tmp[:, 1]
    ptsBase1[:, 1] = tmp[:, 0]
    ptsProj1 = np.loadtxt("./1-PartieManuelle/Serie1/pts_serie1/pts1_12.txt", delimiter=",")
    tmp = np.copy(ptsProj1)
    ptsProj1[:, 0] = tmp[:, 1]
    ptsProj1[:, 1] = tmp[:, 0]
    ptsBase2 = np.loadtxt("./1-PartieManuelle/Serie1/pts_serie1/pts2_32.txt", delimiter=",")
    tmp = np.copy(ptsBase2)
    ptsBase2[:, 0] = tmp[:, 1]
    ptsBase2[:, 1] = tmp[:, 0]
    ptsProj2 = np.loadtxt("./1-PartieManuelle/Serie1/pts_serie1/pts3_32.txt", delimiter=",")
    tmp = np.copy(ptsProj2)
    ptsProj2[:, 0] = tmp[:, 1]
    ptsProj2[:, 1] = tmp[:, 0]
    
    h = calculHomographie(ptsBase1, ptsProj1)

    imH = meth.appliqueTransformation(img1,h)
    plt.imshow(imH)
    plt.show()
