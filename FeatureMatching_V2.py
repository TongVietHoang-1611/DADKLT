import numpy as np
import cv2 as cv2

class FeatureMatching:
    def __init__(self, pattern_path, image_path, ratio_test_threshold=0.75):
        self.pattern_path = pattern_path
        self.image_path = image_path
        self.ratio_test_threshold = ratio_test_threshold


        self.img1 = cv2.cvtColor(self.pattern_path, cv2.COLOR_BGR2GRAY)
        if self.img1 is None:
            raise ValueError("Pattern image loading failed. Check your pattern path.")

  
        self.detector = cv2.SIFT_create()

    def detect_and_compute(self, img):
        """Tìm keypoints và descriptors cho ảnh."""
        kp, des = self.detector.detectAndCompute(img, None)
        return kp, des

    def flann_match_and_draw_polygon(self, kp1, des1, kp2, des2):
        """Sử dụng FLANN để so khớp descriptors và trả về tọa độ polygon."""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_test_threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = self.img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                return dst  
            else:
                print("Homography calculation failed.")
                return None  
        else:
            print("Not enough good matches found.")
            return None  

    def run(self):
        """Thực hiện toàn bộ quy trình và vẽ polygon lên ảnh."""
        # Tìm keypoints và descriptors cho ảnh mẫu
        kp1, des1 = self.detect_and_compute(self.img1)

        #img2 = cv2.imread(self.image_path)
        if self.image_path is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        kp2, des2 = self.detect_and_compute(cv2.cvtColor(self.image_path, cv2.COLOR_BGR2GRAY))
        

        polygon_coords = self.flann_match_and_draw_polygon(kp1, des1, kp2, des2)
        

        if polygon_coords is not None:
            return polygon_coords  
        else:
            return None  