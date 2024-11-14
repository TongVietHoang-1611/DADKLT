import numpy as np
import cv2 as cv2

class FeatureMatching:
    def __init__(self, pattern_path, image_path, ratio_test_threshold=0.75):
        self.pattern_path = pattern_path
        self.image_path = image_path
        self.ratio_test_threshold = ratio_test_threshold

        # Đọc ảnh mẫu dưới dạng ảnh xám
        
        self.img1 = cv2.cvtColor(self.pattern_path, cv2.COLOR_BGR2GRAY)
        if self.img1 is None:
            raise ValueError("Pattern image loading failed. Check your pattern path.")

        # Khởi tạo bộ phát hiện đặc trưng SIFT
        self.detector = cv2.SIFT_create()

    def detect_and_compute(self, img):
        """Tìm keypoints và descriptors cho ảnh."""
        kp, des = self.detector.detectAndCompute(img, None)
        return kp, des

    def flann_match_and_draw_polygon(self, kp1, des1, kp2, des2, img_display):
        """Sử dụng FLANN để so khớp descriptors và bao khung vùng tìm thấy bằng polygon."""
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
                
                img_with_polygon = cv2.polylines(img_display, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                
              #  print(dst)
                
                return img_with_polygon  # Trả về ảnh đã vẽ polygon
            else:
                print("Homography calculation failed.")
                return img_display  # Trả về ảnh gốc nếu không tính toán được homography
        else:
            print("Not enough good matches found.")
            return img_display  # Trả về ảnh gốc nếu không có đủ matches

    def run(self):
        """Thực hiện toàn bộ quy trình."""
        # print("Đang sử dụng thuật toán: SIFT")
        # print("Đang sử dụng matcher: FLANN")

        # Tìm keypoints và descriptors cho ảnh mẫu
        kp1, des1 = self.detect_and_compute(self.img1)


        img2 = cv2.imread(self.image_path)
        if img2 is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        kp2, des2 = self.detect_and_compute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
        img_display = img2.copy()  
        img_with_polygon = self.flann_match_and_draw_polygon(kp1, des1, kp2, des2, img_display)

        return img_with_polygon  
