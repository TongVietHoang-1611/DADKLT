from EasyROI import EasyROI
import cv2
import os

if __name__ == '__main__':
    # Đường dẫn đến ảnh
    image_path = "F:\\DADKLT\\GUI\\ImageForPattern\\raw\\001.jpg"
    folder_path = "F:\\DADKLT\\GUI\\ImageForPattern\\raw"
    
    # Đọc ảnh
    frame = cv2.imread(image_path)
    assert frame is not None, 'Cannot open image'

    # Khởi tạo EasyROI
    roi_helper = EasyROI(verbose=True)

    # Vẽ ROI dưới dạng polygon
    polygon_roi = roi_helper.draw_polygon(frame, 1)

    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    current_index = 0
    
    while True:
    # Đọc ảnh hiện tại
      frame = cv2.imread(image_paths[current_index])
      if frame is None:
          print(f"Không thể đọc ảnh {image_paths[current_index]}")
          break

      # Vẽ ROI lên ảnh
      frame_temp = roi_helper.visualize_roi(frame, polygon_roi)
      cv2.imshow("frame", frame_temp)

      # Chờ nhấn phím
      key = cv2.waitKey(0) & 0xFF
      
      if key == ord('c'):  # Nhấn 'c' để chuyển sang ảnh tiếp theo
          current_index = (current_index + 1) % len(image_paths)
      elif key == ord('q'):  # Nhấn 'q' để thoát
          break

cv2.destroyAllWindows()
    
    
