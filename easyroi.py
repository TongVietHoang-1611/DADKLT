from EasyROI import EasyROI
import cv2
from pprint import pprint

if __name__ == '__main__':
    # Đường dẫn đến ảnh
    image_path = "F:\\DADKLT\\GUI\\ImageForPattern\\raw\\001.jpg"

    # Đọc ảnh
    frame = cv2.imread(image_path)
    assert frame is not None, 'Cannot open image'

    # Khởi tạo EasyROI
    roi_helper = EasyROI(verbose=True)

    # Vẽ ROI dưới dạng polygon
    polygon_roi = roi_helper.draw_polygon(frame, 8)
    print("Polygon Example:")
    pprint(polygon_roi)

    # Lấy thông tin tọa độ của tất cả vertices
    # all_vertices = []
    # for roi_id, roi_data in polygon_roi['roi'].items():
    #     vertices = roi_data['vertices']
    #     all_vertices.append(vertices)
    #     print(f"ROI {roi_id}: Vertices = {vertices}")

    # Hiển thị ảnh với ROI
    frame_temp = roi_helper.visualize_roi(frame, polygon_roi)
    cv2.imshow("frame", frame_temp)
    
    # def display_cropped_images(cropped_images):
    #     for index, cropped_img in cropped_images.items():
    #         window_name = f"Cropped Image {index}"
    #         cv2.imshow(window_name, cropped_img)
    
    # Wait until any key is pressed, then close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # cropped_images = roi_helper.crop_roi(frame, polygon_roi)
    # display_cropped_images(cropped_images)
    
    # Nhấn 'q' để thoát
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
