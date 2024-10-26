import cv2 
from EasyROI import EasyROI

from  FeatureMatching  import  FeatureMatching


def get_cropped_images(cropped_images_dict):
    # Trả về một từ điển chứa các ảnh đã cắt mà không hiển thị
    return {index: cropped_img for index, cropped_img in cropped_images_dict.items()}



if __name__ == '__main__':
    
    image_path = "F:\\DADKLT\\GUI\\ImageForPattern\\raw\\001.jpg"
    image_path_test = "F:\\DADKLT\\GUI\\ImageForPattern\\raw\\005.jpg"
    frame = cv2.imread(image_path)
    assert frame is not None, 'Cannot open image'
    
    roi_helper = EasyROI(verbose=True)
    
    rectangle_roi=roi_helper.draw_rectangle(frame, 1)
    frame_temp = roi_helper.visualize_roi(frame, rectangle_roi)
    polygon_roi=roi_helper.draw_polygon(frame_temp, 1)
    # frame_final =roi_helper.visualize_roi(frame_temp, polygon_roi)
    #cv2.imshow("frame",  frame_final)
    
    cropped_image_rectangle = roi_helper.crop_roi(frame, rectangle_roi)
    cropped_images = get_cropped_images(cropped_image_rectangle)
    for index, cropped_img in cropped_images.items():
        window_name = f"Cropped Image {index}"
        #cv2.imshow(window_name, cropped_img)
        
        feature_matching=FeatureMatching(cropped_img, image_path_test)
        
        feature_matching.run()
    
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()