import numpy as np
import imageio as io
import rechauffement as meth
import matplotlib.pyplot as plt

# Création d'une matrice Nx3 pour les points de bases
def matriceNx3(ptsBase):
    A = []
    n = ptsBase.shape[0]
    for i in range(0,n):
        pts = np.zeros((1,3), dtype=int)
        # point i
        np.put(pts, [0,1], ptsBase[i])
        np.put(pts, [2], 1)
        A.append(pts)
    A = np.reshape(A, (n,3))
    return A

# Création d'une matrice pour trouver la transformation projective
# Places les points entre deux images dans une matrice 2*Nx9
def matriceResolve(ptsBase, ptsProj):
    A = []
    n = ptsBase.shape[0]
    for i in range(0,n):
        pts = np.zeros((2,9), dtype=int)
        # points i
        np.put(pts, [0,1,2], -1*(ptsBase[i]))
        np.put(pts, [6,7,8], ptsBase[i]*ptsProj[i,0])
        np.put(pts, [12,13,14], -1*(ptsBase[i]))
        np.put(pts, [15,16,17], ptsBase[i]*ptsProj[i,1])
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

# # Création d'une image commune (Mosaïque)
# def mosaique(img1, img2, img3):


if __name__ == "__main__":
    np.set_printoptions(precision=3,suppress=True)

    # Extraction des images
    img1 = io.imread('./1-PartieManuelle/Serie1/IMG_2415.jpg')
    img2 = io.imread('./1-PartieManuelle/Serie1/IMG_2416.jpg')
    img3 = io.imread('./1-PartieManuelle/Serie1/IMG_2417.jpg')

    # img1 = io.imread('./1-PartieManuelle/Serie2/IMG_2427.jpg')
    # img2 = io.imread('./1-PartieManuelle/Serie2/IMG_2426.jpg')
    # img3 = io.imread('./1-PartieManuelle/Serie2/IMG_2425.jpg')

    # img1 = io.imread('./1-PartieManuelle/Serie3/IMG_2409.jpg')
    # img2 = io.imread('./1-PartieManuelle/Serie3/IMG_2410.jpg')
    # img3 = io.imread('./1-PartieManuelle/Serie3/IMG_2411.jpg')

    # Extraction des points des images
    ptsBase1 = np.loadtxt("./1-PartieManuelle/Serie1/pts_serie1/pts2_12.txt", delimiter=",")
    ptsProj1 = np.loadtxt("./1-PartieManuelle/Serie1/pts_serie1/pts1_12.txt", delimiter=",")
    ptsBase2 = np.loadtxt("./1-PartieManuelle/Serie1/pts_serie1/pts2_32.txt", delimiter=",")
    ptsProj2 = np.loadtxt("./1-PartieManuelle/Serie1/pts_serie1/pts3_32.txt", delimiter=",")

    # ptsBase1 = np.loadtxt("./1-PartieManuelle/Serie2/pts_serie2/pts2_12.txt", delimiter=",")
    # ptsProj1 = np.loadtxt("./1-PartieManuelle/Serie2/pts_serie2/pts1_12.txt", delimiter=",")
    # ptsBase2 = np.loadtxt("./1-PartieManuelle/Serie2/pts_serie2/pts2_32.txt", delimiter=",")
    # ptsProj2 = np.loadtxt("./1-PartieManuelle/Serie2/pts_serie2/pts3_32.txt", delimiter=",")

    # ptsBase1 = np.loadtxt("./1-PartieManuelle/Serie3/pts_serie3/pts2_12.txt", delimiter=",")
    # ptsProj1 = np.loadtxt("./1-PartieManuelle/Serie3/pts_serie3/pts1_12.txt", delimiter=",")
    # ptsBase2 = np.loadtxt("./1-PartieManuelle/Serie3/pts_serie3/pts2_32.txt", delimiter=",")
    # ptsProj2 = np.loadtxt("./1-PartieManuelle/Serie3/pts_serie3/pts3_32.txt", delimiter=",")
    
    # Calcul de l'homographie
    H1 = calculHomographie(ptsProj1, ptsBase1)
    H3 = calculHomographie(ptsProj2, ptsBase2)

    # Appliquer l'homographie 1 à l'image 1
    # imH1 = meth.appliqueTransformation(img1,H1)
    # io.imwrite('S2H1.jpg', imH1)
    # plt.imshow(imH1)
    # plt.show()
    imH1 = io.imread('./1-PartieManuelle/Serie1/S1H1.jpg')
    # imH1 = io.imread('./1-PartieManuelle/Serie2/S2H1.jpg')
    # imH1 = io.imread('./1-PartieManuelle/Serie3/S3H1.jpg')
    print(imH1.shape)

    # Appliquer l'homographie 3 à l'image 3
    # imH3 = meth.appliqueTransformation(img3,H3)
    # io.imwrite('S2H3.jpg', imH3)
    # plt.imshow(imH3)
    # plt.show()
    imH3 = io.imread('./1-PartieManuelle/Serie1/S1H3.jpg')
    # imH3 = io.imread('./1-PartieManuelle/Serie2/S2H3.jpg')
    # imH3 = io.imread('./1-PartieManuelle/Serie3/S3H3.jpg')
    print(imH3.shape)

    # Calcul du coin d'origine
    origin1 = np.matmul(H1,[0,0,1])/np.matmul(H1,[0,0,1])[2]
    x1 = abs(origin1[0]).astype(int)
    y1 = abs(origin1[1]).astype(int)
    print(origin1)
    origin2 = np.matmul(H3,[0,0,1])/np.matmul(H3,[0,0,1])[2]
    x2 = abs(origin2[0]).astype(int)
    y2 = abs(origin2[1]).astype(int)
    print(origin2)

    
    imF = np.zeros((max(imH1.shape[0],img2.shape[0],imH3.shape[0])+y1+y2, imH1.shape[1]+img2.shape[1]+imH3.shape[1]-(x1+x2), 3), dtype=int)

    # Image serie 1
    imF[:imH1.shape[0],:imH1.shape[1],:] = imH1
    imF[y2+57:imH3.shape[0]+y2+57,x2+x1:imH3.shape[1]+x2+x1,:] = imH3
    imF[y1:img2.shape[0]+y1,x1:img2.shape[1]+x1,:] = img2
    io.imwrite('S1F.jpg', imF)
    plt.imshow(imF)
    plt.show()

    # # Image serie 2
    # imF[:imH1.shape[0],:imH1.shape[1],:] = imH1
    # imF[y2+57:imH3.shape[0]+y2+57,x2+x1:imH3.shape[1]+x2+x1,:] = imH3
    # imF[y1:img2.shape[0]+y1,x1:img2.shape[1]+x1,:] = img2
    # io.imwrite('S2F.jpg', imF)
    # plt.imshow(imF)
    # plt.show()

    # # Image serie 3
    # imF[:imH1.shape[0],:imH1.shape[1],:] = imH1
    # imF[y2+57:imH3.shape[0]+y2+57,x2+x1:imH3.shape[1]+x2+x1,:] = imH3
    # imF[y1:img2.shape[0]+y1,x1:img2.shape[1]+x1,:] = img2
    # io.imwrite('S3F.jpg', imF)
    # plt.imshow(imF)
    # plt.show()