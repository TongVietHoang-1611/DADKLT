import cv2
import os
import numpy as np
from EasyROI import EasyROI
from FeatureMatching_V2 import FeatureMatching

def take_polygon_roi(data):
    result = {}
    for key, roi_data in data['roi'].items():
        vertices = roi_data['vertices']
        if len(vertices) >= 4:
            result[key] = vertices[:4]
    return result

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

if __name__ == '__main__':
    # Định nghĩa đường dẫn
    image_path = "F:\\DADKLT\\GUI\\ImageForPattern\\raw\\001.jpg"
    folder_path = "F:\\DADKLT\\GUI\\ImageForPattern\\raw"
    image_paths = get_image_paths(folder_path)
    
    # Đọc ảnh đầu tiên
    frame = cv2.imread(image_path)
    assert frame is not None, 'Cannot open image'

    # Khởi tạo ROI helper và vẽ các ROI
    roi_helper = EasyROI(verbose=True)
    rectangle_roi = roi_helper.draw_rectangle(frame, 1)
    frame_temp = roi_helper.visualize_roi(frame, rectangle_roi)
    polygon_roi = roi_helper.draw_polygon(frame_temp, 2)
    
    # Xử lý ảnh và lấy các thông tin ban đầu
    cropped_image_rectangle = roi_helper.crop_roi(frame, rectangle_roi)
    cropped_images = get_cropped_images(cropped_image_rectangle)
    init_rectangle = get_rect_roi(rectangle_roi)[0]  # Lấy rectangle đầu tiên
    init_polygon = take_polygon_roi(polygon_roi)
    
    # Xử lý từng ảnh trong thư mục
    current_index = 0
    while current_index < len(image_paths):
        image_path_test = image_paths[current_index]
        frame_test = cv2.imread(image_path_test)
        assert frame_test is not None, f'Cannot open image: {image_path_test}'
        
        for index, cropped_img in cropped_images.items():
            # Thực hiện feature matching
            feature_matching = FeatureMatching(cropped_img, image_path_test)          
            new_rectangle_roi = np.int32(feature_matching.run())
            
            # Áp dụng biến đổi affine để cập nhật polygon
            new_polygon_roi = transform_polygon_roi(init_polygon, init_rectangle, new_rectangle_roi)
            
            # Hiển thị kết quả
            frame_final = visualize_transformation(frame_test, new_rectangle_roi, new_polygon_roi)
            cv2.imshow("Matched Region", frame_final)

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