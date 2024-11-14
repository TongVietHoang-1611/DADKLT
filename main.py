import cv2
import os
from EasyROI import EasyROI
from FeatureMatching_V2 import FeatureMatching
import numpy as np

def take_polygon_roi(data):
    result = {}
    for key, roi_data in data['roi'].items():
        vertices = roi_data['vertices']
        if len(vertices) >= 4:
            result[key] = vertices[:4]
    return result

def get_image_paths(folder_path):
    """Trả về danh sách đường dẫn tới các ảnh trong thư mục."""
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

def get_cropped_images(cropped_images_dict):
    """Trả về dictionary chứa các ảnh đã cắt."""
    return {index: cropped_img for index, cropped_img in cropped_images_dict.items()}

def get_polygon_coordinates(roi_dict):
    """Lấy tọa độ các đỉnh của polygon ROI."""
    polygons_coordinates = {}
    for index, roi in roi_dict['roi'].items():
        poly_vertices = [(int(vertex[0]), int(vertex[1])) for vertex in roi['vertices']]
        polygons_coordinates[index] = poly_vertices
    return polygons_coordinates

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

def calculate_center(rectangle_points):
    # Lấy tọa độ từ danh sách
    x1, y1 = rectangle_points[0][0][0]
    x2, y2 = rectangle_points[0][1][0]
    x3, y3 = rectangle_points[0][2][0]
    x4, y4 = rectangle_points[0][3][0]
    
    # Tính tọa độ trung tâm
    center_x = (x1 + x3) / 2
    center_y = (y1 + y3) / 2
    
    return (int(center_x), int(center_y))

def calculate_center_pattern(rectangle_points):
    # Lấy tọa độ từ danh sách mà không cần giải nén
    x1, y1 = rectangle_points[0][0]
    x2, y2 = rectangle_points[1][0]
    x3, y3 = rectangle_points[2][0]
    x4, y4 = rectangle_points[3][0]
    
    # Tính tọa độ trung tâm
    center_x = (x1 + x3) / 2
    center_y = (y1 + y3) / 2
    
    return (int(center_x), int(center_y))

def calculate_change(center1, center2):
    # Tính sự thay đổi giữa hai tâm
    x1, y1 = center1
    x2, y2 = center2
    
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    return delta_x, delta_y

def convert_to_polygon_format(roi_dict):
    result = {'type': 'polygon', 'roi': {}}
    
    for key, points in roi_dict.items():
        # Tạo danh sách vertices với np.int32 và làm tròn giá trị
        vertices = [(np.int32(int(round(x))), np.int32(int(round(y)))) for x, y in points]
        
        # Cập nhật roi_dict với thông tin vertices
        result['roi'][key] = {'vertices': vertices}
    
    return result

def apply_change_to_polygon(original_polygon, delta_x, delta_y):
    new_roi_dict = {}
    # Tạo bản sao của polygon gốc để tránh thay đổi giá trị gốc
    for key, polygon_points in original_polygon.items():
        new_polygon = []
        for point in polygon_points:
            x, y = point
            # Cộng sự thay đổi vào tọa độ
            new_x = x + delta_x
            new_y = y + delta_y
            new_polygon.append((new_x, new_y))
        new_roi_dict[key] = new_polygon
    
    return convert_to_polygon_format(new_roi_dict)

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
    init_cordinate = get_rect_roi(rectangle_roi)
    init_polygon = take_polygon_roi(polygon_roi)
    init_center = calculate_center(init_cordinate)
    
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
            
            # Tính toán sự thay đổi vị trí
            new_center = calculate_center_pattern(new_rectangle_roi)
            center_change = calculate_change(init_center, new_center)
            
            # Áp dụng sự thay đổi lên polygon ban đầu
            new_polygon_roi = apply_change_to_polygon(init_polygon, center_change[0], center_change[1])
            
            # Hiển thị kết quả
            img_display = cv2.imread(image_path_test)
            img_with_polygon = cv2.polylines(img_display, [new_rectangle_roi], True, (0, 255, 0), 3, cv2.LINE_AA)
            frame_final = roi_helper.visualize_roi(img_with_polygon, new_polygon_roi)
            cv2.imshow("Matched Region", frame_final)

        # Xử lý phím nhấn
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c'):  # Nhấn 'c' để chuyển sang ảnh tiếp theo
            current_index += 1
            if current_index >= len(image_paths):
                print("Đã tới ảnh cuối cùng.")
                break
        elif key == ord('q'):  # Nhấn 'q' để thoát
            break

    cv2.destroyAllWindows()