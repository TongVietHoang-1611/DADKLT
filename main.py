import cv2
import os
import numpy as np
from EasyROI import EasyROI
from FeatureMatching_V2 import FeatureMatching
import time
import zxingcpp
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en", use_gpu=False)

def take_polygon_roi(data):
    result = {}
    for key, roi_data in data['roi'].items():
        vertices = roi_data['vertices']
        if len(vertices) >= 4:
            result[key] = vertices[:4]
    return result

def crop_polygon_roi(image, polygon_roi):
    """
    Cắt các vùng ROI đa giác từ ảnh dựa trên polygon_roi.
    :param image: Ảnh gốc
    :param polygon_roi: Định dạng polygon ROI {'type': 'polygon', 'roi': {0: {'vertices': [...]}, ...}}
    :return: Dictionary chứa các vùng đã được cắt
    """
    cropped_regions = {}

    for key, roi_data in polygon_roi['roi'].items():
        # Lấy vertices của ROI
        vertices = np.array(roi_data['vertices'], dtype=np.int32)

        # Tạo mask với vùng đa giác
        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Mask có kích thước giống ảnh, dạng grayscale
        cv2.fillPoly(mask, [vertices], 255)  # Vẽ polygon lên mask

        # Áp dụng mask lên ảnh để lấy vùng ROI
        cropped = cv2.bitwise_and(image, image, mask=mask)

        # Cắt vừa khung bounding box của polygon để giảm kích thước
        x, y, w, h = cv2.boundingRect(vertices)  # Tạo bounding box từ polygon
        cropped = cropped[y:y+h, x:x+w]  # Cắt vùng bounding box

        cropped_regions[key] = cropped  # Lưu lại vùng cắt với key tương ứng

    return cropped_regions

def get_image_paths(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

def get_cropped_images(cropped_images_dict):
    return {index: cropped_img for index, cropped_img in cropped_images_dict.items()}

def get_rect_roi(roi_dict):
    result = []
    for index, roi in roi_dict['roi'].items():
        tl_x = roi['tl_x']
        tl_y = roi['tl_y']
        br_x = roi['br_x']
        br_y = roi['br_y']

        four_points = [
            [[tl_x, tl_y]],      # Góc trên-trái
            [[tl_x, br_y]],      # Góc dưới-trái
            [[br_x, br_y]],      # Góc dưới-phải
            [[br_x, tl_y]]       # Góc trên-phải
        ]
        result.append(four_points)
    return result

def convert_to_polygon_format(roi_dict):
    result = {'type': 'polygon', 'roi': {}}
    for key, points in roi_dict.items():
        vertices = [(np.int32(int(round(x))), np.int32(int(round(y)))) for x, y in points]
        result['roi'][key] = {'vertices': vertices}
    return result

def get_affine_transform(src_points, dst_points):
    """
    Tính ma trận biến đổi affine từ các điểm nguồn đến các điểm đích
    """
    src_points = np.float32([point[0] for point in src_points[:3]])
    dst_points = np.float32([point[0] for point in dst_points[:3]])
    return cv2.getAffineTransform(src_points, dst_points)

def apply_affine_transform_to_polygon(polygon_points, affine_matrix):
    """
    Áp dụng phép biến đổi affine lên các điểm của polygon
    """
    transformed_points = []
    for point in polygon_points:
        x, y = point
        # Chuyển đổi điểm thành ma trận homogeneous
        point_matrix = np.array([x, y, 1])
        # Áp dụng phép biến đổi
        transformed_point = np.dot(affine_matrix, point_matrix)
        transformed_points.append((int(transformed_point[0]), int(transformed_point[1])))
    return transformed_points

def transform_polygon_roi(original_polygon, init_rectangle, new_rectangle):
    """
    Biến đổi polygon dựa trên sự thay đổi của rectangle
    """
    # Tính ma trận biến đổi affine
    affine_matrix = get_affine_transform(init_rectangle, new_rectangle)
    
    new_roi_dict = {}
    for key, polygon_points in original_polygon.items():
        # Áp dụng biến đổi affine lên từng điểm của polygon
        transformed_points = apply_affine_transform_to_polygon(polygon_points, affine_matrix)
        new_roi_dict[key] = transformed_points
    
    return convert_to_polygon_format(new_roi_dict)

def visualize_transformation(img, rectangle_points, polygon_roi, title="Transformation"):
    """
    Hiển thị kết quả transformation
    """
    # Vẽ rectangle
    img_with_rect = cv2.polylines(img.copy(), [rectangle_points], True, (0, 255, 0), 2)
    
    # Vẽ polygon
    for key, roi in polygon_roi['roi'].items():
        vertices = np.array(roi['vertices'], np.int32)
        img_with_rect = cv2.polylines(img_with_rect, [vertices], True, (0, 0, 255), 2)
    
    return img_with_rect
def crop_left_half(image):
    """
    Cắt nửa bên trái của bức ảnh.
    """
    height, width = image.shape[:2]
    left_half = image[:, :width // 2]
    return left_half

def resize_image(image, scale_percent=80):
    """
    Resize ảnh theo tỷ lệ phần trăm.
    :param image: Ảnh đầu vào
    :param scale_percent: Tỷ lệ resize (mặc định 80%)
    :return: Ảnh đã được resize
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

if __name__ == '__main__':
    # Định nghĩa đường dẫn
    image_path = "F:\\DADKLT\\GUI\\WIN_20241122_21_17_58_Pro.jpg"
    folder_path = "F:\\DADKLT\\GUI\\ImageForPattern\\test"
    image_paths = get_image_paths(folder_path)
    
    # Đọc ảnh đầu tiên và resize
    frame = cv2.imread(image_path)
    assert frame is not None, 'Cannot open image'
    frame = resize_image(frame, scale_percent=80)  # Resize ảnh đầu vào xuống 80%

    # Khởi tạo ROI helper và vẽ các ROI
    roi_helper = EasyROI(verbose=True)
    
    rectangle_roi = roi_helper.draw_rectangle(frame, 1)
    frame_temp = roi_helper.visualize_roi(frame, rectangle_roi)
    polygon_roi = roi_helper.draw_polygon(frame_temp, 6)
    
    # Xử lý ảnh và lấy các thông tin ban đầu
    cropped_image_rectangle = roi_helper.crop_roi(frame, rectangle_roi)
    cropped_images = get_cropped_images(cropped_image_rectangle)
    init_rectangle = get_rect_roi(rectangle_roi)[0]  # Lấy rectangle đầu tiên
    init_polygon = take_polygon_roi(polygon_roi)
    
    # Xử lý từng ảnh trong thư mục
    current_index = 0
    while current_index < len(image_paths):
        image_path_test = image_paths[current_index]
        frame_test = cv2.imread(image_path_test)  # Đọc ảnh test
        assert frame_test is not None, f'Cannot open image: {image_path_test}'
        frame_test = resize_image(frame_test, scale_percent=80)  # Resize ảnh test xuống 80%
        
        cropped_frame_test = crop_left_half(frame_test)  # Chỉ dùng ảnh này để tìm pattern
        
        start_time = time.time()  # Bắt đầu đo thời gian xử lý
        
        for index, cropped_img in cropped_images.items():
            # Thực hiện feature matching trên nửa trái
            feature_matching = FeatureMatching(cropped_img, cropped_frame_test)
            new_rectangle_roi = np.int32(feature_matching.run())  # run() cần xử lý input là ma trận NumPy
            
            # Áp dụng biến đổi affine để cập nhật polygon
            new_polygon_roi = transform_polygon_roi(init_polygon, init_rectangle, new_rectangle_roi)
            
            # print(new_polygon_roi)
            

            
            cropped_regions = crop_polygon_roi(frame_test, new_polygon_roi)
            last_cropped_key = max(cropped_regions.keys())
            last_cropped_image = cropped_regions[last_cropped_key]
            

            
            results = zxingcpp.read_barcodes(last_cropped_image)
            
            print("Decoded Data Matrix values:")
            for result in results:
                print("{}"
                    .format(result.text))
            batch_images = []
            for key, region in cropped_regions.items():
                if key != last_cropped_key:
                    resized_region = resize_image(region, scale_percent=70)
                    batch_images.append(resized_region)
            if batch_images:
                ocr_results = ocr.ocr(batch_images, det=False)
                for idx, result in enumerate(ocr_results):
                    print(f"OCR Results for Region {idx + 1}:")
                    for line in result:
                        print(line)

            
            for key, region in cropped_regions.items():
                cv2.imshow(f"ROI {key}", region)  # Hiển thị vùng đã cắt
            
            # Hiển thị kết quả trên toàn bộ ảnh test
            frame_final = visualize_transformation(frame_test.copy(), new_rectangle_roi, new_polygon_roi)
            cv2.imshow("Matched Region on Full Image", frame_final)
        
        # Kết thúc đo thời gian xử lý
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Thời gian xử lý ảnh {current_index + 1}: {processing_time:.2f} giây")
        
        # Xử lý phím nhấn
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c'):
            current_index += 1
            if current_index >= len(image_paths):
                print("Đã tới ảnh cuối cùng.")
                break
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


