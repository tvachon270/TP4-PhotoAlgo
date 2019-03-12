import numpy as np
import imageio as io

# Création d'une matrice 4x3 pour les points de bases
def matrice4x3(ptsBase):
    A = np.zeros((4,3), dtype=int)
    # point 1
    np.put(A, [0,1], ptsBase[0])
    np.put(A, [2], 1)
    # point 2
    np.put(A, [3,4], ptsBase[1])
    np.put(A, [5], 1)
    # point 3
    np.put(A, [6,7], ptsBase[2])
    np.put(A, [8], 1)
    # point 4
    np.put(A, [9,10], ptsBase[3])
    np.put(A, [11], 1)
    return A

# Création d'une matrice pour trouver la transformation projective
# Places les points entre deux images dans une matrice 8x9
def matriceResolve(imgBase, imgProj):
    A = np.zeros((8,9), dtype=int)
    # points 1
    np.put(A, [0,1,2], -1*(imgBase[0]))
    np.put(A, [6,7,8], imgBase[0]*imgProj[0][0])
    np.put(A, [12,13,14], -1*(imgBase[0]))
    np.put(A, [15,16,17], imgBase[0]*imgProj[0][1])
    # points 2
    np.put(A, [18,19,20], -1*(imgBase[1]))
    np.put(A, [24,25,26], imgBase[1]*imgProj[1][0])
    np.put(A, [30,31,32], -1*(imgBase[1]))
    np.put(A, [33,34,35], imgBase[1]*imgProj[1][1])
    # points 3
    np.put(A, [36,37,38], -1*(imgBase[2]))
    np.put(A, [42,43,44], imgBase[2]*imgProj[2][0])
    np.put(A, [48,49,50], -1*(imgBase[2]))
    np.put(A, [51,52,53], imgBase[2]*imgProj[2][1])
    # points 4
    np.put(A, [54,55,56], -1*(imgBase[3]))
    np.put(A, [60,61,62], imgBase[3]*imgProj[3][0])
    np.put(A, [66,67,68], -1*(imgBase[3]))
    np.put(A, [69,70,71], imgBase[3]*imgProj[3][1])
    return A


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


# Création de la matrice A
ptsBase1 = matrice4x3(ptsBase1)
A = matriceResolve(ptsBase1, ptsProj1)


# Décomposition en valeur singulière
U,S,V = np.linalg.svd(np.transpose(A))
print("U")
print(U.shape)
print("S")
print(S.shape)
print("V")
print(V.shape)