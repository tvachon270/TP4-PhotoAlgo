import cv2
import numpy as np
import imageio as io
import harris as hs
import rechauffement as re
import appariementManuel as am
import matplotlib.pyplot as plt

def descripteurs(image1, pts1):
    desc = []
    j = 0
    for i in range(0, pts1.shape[0]):
        if pts1[i][0] - 20 > 0 and pts1[i][0] + 20 < image1.shape[0] and pts1[i][1] - 20 > 0 and pts1[i][1] + 20 < image1.shape[1]:
            j = j+1
            desc.append(np.ravel(cv2.resize(image1[pts1[i][0]-20:pts1[i][0]+20, pts1[i][1]-20:pts1[i][1]+20], (8, 8))))
    desc = np.float32(np.reshape(desc, (j,64)))
    return desc
        
def fits(des1, des2):
    MIN_MATCH_COUNT = 10
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return good

def findPts(pts1, pts2, fits):
    pt_src = []
    pt_dst = []
    for i in range(0,len(fits)):
        pt_src.append(pts1[fits[i].queryIdx])
        pt_dst.append(pts2[fits[i].trainIdx])
    pt_src = np.reshape(pt_src, (len(fits),2))
    pt_dst = np.reshape(pt_dst, (len(fits),2))
    return pt_src, pt_dst
    # src_pts = np.float32([ pts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    # dst_pts = np.float32([ pts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # return src_pts, dst_pts
            

if __name__ == "__main__":
    img0 = io.imread('./2-PartieAutomatique/Serie1/goldengate-00.png')
    img1 = io.imread('./2-PartieAutomatique/Serie1/goldengate-01.png')
    img2 = io.imread('./2-PartieAutomatique/Serie1/goldengate-02.png')
    img3 = io.imread('./2-PartieAutomatique/Serie1/goldengate-03.png')
    img4 = io.imread('./2-PartieAutomatique/Serie1/goldengate-04.png')
    img5 = io.imread('./2-PartieAutomatique/Serie1/goldengate-05.png')

    pts0 = hs.harris2('./2-PartieAutomatique/Serie1/goldengate-00.png')
    pts1 = hs.harris2('./2-PartieAutomatique/Serie1/goldengate-01.png')
    pts2 = hs.harris2('./2-PartieAutomatique/Serie1/goldengate-02.png')
    pts3 = hs.harris2('./2-PartieAutomatique/Serie1/goldengate-03.png')
    pts4 = hs.harris2('./2-PartieAutomatique/Serie1/goldengate-04.png')
    pts5 = hs.harris2('./2-PartieAutomatique/Serie1/goldengate-05.png')

    des0 = descripteurs(img0, pts0)
    des1 = descripteurs(img1, pts1)
    des2 = descripteurs(img2, pts2)
    des3 = descripteurs(img3, pts3)
    des4 = descripteurs(img4, pts4)
    des5 = descripteurs(img5, pts5)

    fit01 = fits(des0, des1)
    print(fit01[0].queryIdx)
    print(fit01[0].trainIdx)
    fit12 = fits(des1, des2)
    fit23 = fits(des2, des3)
    fit34 = fits(des3, des4)
    fit45 = fits(des4, des5)

    pt_src, pt_dst = findPts(pts0, pts1, fit01)
    
    
    H0 = am.calculHomographie(pt_dst, pt_src)
    imH0 = re.appliqueTransformation(img0,H0)
    plt.imshow(imH0)
    plt.show()


    #for j in range(0,pts0.shape[0]): resized_images[j][0] = cv2.resize(img0[pts0[0][0]-20:pts0[0][0]+20, pts0[0][1]-20:pts0[0][1]+20], (8, 8))

    # gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    # gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    # gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    