import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Thiết lập giao diện
        self.image_label = QLabel(self)
        self.image_label.setText("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.btn_select = QPushButton("Select Image", self)
        self.btn_select.clicked.connect(self.open_file_dialog)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_select)

        self.setLayout(layout)
        self.setWindowTitle("Image Selector")
        self.setGeometry(300, 300, 400, 300)

    def open_file_dialog(self):
        # Hiển thị hộp thoại chọn file
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_name:
            # Hiển thị ảnh đã chọn
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), aspectRatioMode=True))
            self.image_label.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageSelector()
    ex.show()
    sys.exit(app.exec_())
