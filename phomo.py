import numpy as np

# source  https://github.com/bilaer/Planar-homography/blob/master/Homography.py

###########################################################################
#                         Homography functions                            #
###########################################################################

# Function that use SVD to calculate the homography matrix
# correspondences must be at least four pair points
# [[(x1, y1),(x2, y2)]...]
# this function normalize the homography matrixso that the
# solutiion is more stable The transformation matrix is given by
# T = s*[[1, 0, - mean(u)], [0, 1, -mean(v)], [0, 0, 1/s]] where
# X1 is before transformation, X2 is after transformation
# s is given by s = 2**0.5*n/sum((ui - mean(u)**2 + (vi - mean(v))**2)**0.5
# reference: http://www.ele.puc-rio.br/~visao/Homographies.pdf


def calHomography(correspondences):
    assert (len(correspondences) >= 4)
    meanX1, meanY1, meanX2, meanY2 = 0, 0, 0, 0
    for point in correspondences:
        meanX1 = point[0][0] + meanX1
        meanX2 = point[1][0] + meanX2
        meanY1 = point[0][1] + meanY1
        meanY2 = point[1][1] + meanY2
    meanX1, meanY1 = meanX1 / \
        len(correspondences), meanY1 / len(correspondences)
    meanX2, meanY2 = meanX2 / \
        len(correspondences), meanY2 / len(correspondences)

    # Calculate the scale factor
    s1, s2 = 0, 0
    for point in correspondences:
        s1 = s1 + ((point[0][0] - meanX1)**2 + (point[0][1] - meanY1)**2)**0.5
        s2 = s2 + ((point[1][0] - meanX2)**2 + (point[1][1] - meanY2)**2)**0.5
    s1 = (2**0.5)*len(correspondences)/s1
    s2 = (2**0.5)*len(correspondences)/s2

    # Get the transformation matrix
    T1 = s1 * np.array([[1, 0, -meanX1], [0, 1, -meanY1], [0, 0, 1 / s1]])
    T2 = s2 * np.array([[1, 0, -meanX2], [0, 1, -meanY2], [0, 0, 1 / s2]])

    # Calculate the homography matrix of normalized the coordinates
    A = []
    for i in range(len(correspondences)):
        # correspondences has form [[(x, y, level),(x, y, level)]....]
        norm1 = np.array([[correspondences[i][0][0]], [
                         correspondences[i][0][1]], [1]])
        norm2 = np.array([[correspondences[i][1][0]], [
                         correspondences[i][1][1]], [1]])
        p1 = np.matmul(T1, norm1)
        p2 = np.matmul(T2, norm2)
        x1, y1, x2, y2 = p1[0][0], p1[1][0], p2[0][0], p2[1][0]
        A = A + [[0, 0, 0, -x2, -y2, -1,  x2*y1, y2*y1, y1],
                 [x2, y2, 1, 0, 0, 0, -x2*x1, -y2*x1, -x1]]

    # Get transpose A multiple A
    A = np.array(A)
    B = np.matmul(np.transpose(A), A)
    # Get the SVD decomposition of ATA
    u, s, vh = np.linalg.svd(B)  # A = UEV T
    # The solution is the smallest eigenvalue and it is corresponding eigenvector
    # solution has form h1, h2 ... h9 and reshape the vector into a matrix
    H = np.reshape(vh[8, :], (3, 3))
    # Multiple transformation matrix to get true homography
    H = np.matmul(np.matmul(np.linalg.inv(T1), H), T2)

    return H


ranCorr = []
# [X1, Y1], [X2, Y2]
ranCorr.append([(0, 0), (554, 548)])
ranCorr.append([(7, 0), (1074, 537)])
ranCorr.append([(7, 13.2), (1178, 969)])
ranCorr.append([(0.2, 13.2), (192, 993)])
ph = calHomography(ranCorr)
# [X2, Y2]
test00 = np.array([554, 548, 1])
out00 = np.matmul(ph, test00)
test70 = np.array([1074, 537, 1])
out70 = np.matmul(ph, test70)
test13 = np.array([1178, 969, 1])
out13 = np.matmul(ph, test13)
test132 = np.array([192, 993, 1])
out132 = np.matmul(ph, test132)
print("Homography Matrix\n", ph)
print("output 0 0", out00/out00[2])
print("output 7 0", out70/out70[2])
print("output 7 13.2", out13/out13[2])
print("output 0.2 13.2", out132/out132[2])
print("--------------")

# Normalize the homography matrix
normal = ph
normal = normal / normal[2][2]
print("Homography Matrix Normalized\n", normal)
test = np.array([1178, 969, 1])
testn = np.matmul(normal, np.transpose(test))
print("output for input [7 13.2] after normalized\n",
      np.transpose(testn/testn[2]))


# xranCorr = []
# xranCorr.append([(0, 0), (0, 0)])
# xranCorr.append([(1, 0), (100, 0)])
# xranCorr.append([(1, 1), (100, 100)])
# xranCorr.append([(0, 1), (0, 100)])
# xph = calHomography(xranCorr)

# dummy_input = np.array([100, 100, 1])
# out1 = np.matmul(xph, dummy_input)
# print("test homography", xph)
# print("test output", out1/out1[2])
