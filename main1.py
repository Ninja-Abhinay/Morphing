import cv2
import numpy as np
import dlib

# Initialize Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    for face in faces:
        landmarks = [(p.x, p.y) for p in predictor(image, face).parts()]
        return np.array(landmarks, np.int32)
    return np.array([])

def apply_affine_transform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(img1, img2, imgMorph, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2Rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, (r[2], r[3]))
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, (r[2], r[3]))

    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    imgMorph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = imgMorph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask

def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))
    triangleList = subdiv.getTriangleList()
    delaunayTri = []

    for t in triangleList:
        pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        ptInd = []
        for j in range(0, 3):
            for k in range(0, len(points)):
                if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                    ptInd.append(k)
        if len(ptInd) == 3:
            delaunayTri.append((ptInd[0], ptInd[1], ptInd[2]))
    return delaunayTri

def load_and_prepare_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image {image_path}")
        return None
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.astype(np.uint8)

def morph_faces(filename1, filename2, alpha=0.5):
    img1 = load_and_prepare_image(filename1)
    img2 = load_and_prepare_image(filename2)
    if img1 is None or img2 is None:
        return None

    img1_resized, img2_resized = [cv2.resize(img, (max(img1.shape[1], img2.shape[1]), max(img1.shape[0], img2.shape[0]))) for img in [img1, img2]]

    points1 = get_landmarks(img1_resized)
    points2 = get_landmarks(img2_resized)
    points = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(points1, points2)]

    imgMorph = np.zeros(img1_resized.shape, dtype=img1_resized.dtype)

    rect = (0, 0, img1_resized.shape[1], img1_resized.shape[0])
    dt = calculate_delaunay_triangles(rect, points)

    for i in range(len(dt)):
        t1 = [points1[dt[i][0]], points1[dt[i][1]], points1[dt[i][2]]]
        t2 = [points2[dt[i][0]], points2[dt[i][1]], points2[dt[i][2]]]
        t = [points[dt[i][0]], points[dt[i][1]], points[dt[i][2]]]
        
        morph_triangle(img1_resized, img2_resized, imgMorph, t1, t2, t, alpha)

    return imgMorph

if __name__ == "__main__":
    filename1 = "1.png"  # Replace with the path to your first image
    filename2 = "2.png"  # Replace with the path to your second image
    alpha = 0.5  # Set the morphing parameter (0 to 1)
    morphed_image = morph_faces(filename1, filename2, alpha)
    if morphed_image is not None:
        cv2.imshow("Morphed Face", morphed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to morph images.")
