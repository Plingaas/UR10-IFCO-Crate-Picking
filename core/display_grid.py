import numpy as np
import cv2


class DisplayGrid:
    def __init__(self, width=1800, height=800, cols=4, rows=1) -> None:
        self.width = width
        self.height = height
        self.cols = cols
        self.rows = rows
        self.cell_width = width // cols
        self.cell_height = height // rows
        self.display = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)  # Color display
        self.window = cv2.namedWindow("Display")

    def set_data(self, x, y, image):
        """Places image into cell at (x, y)"""
        if image is None:
            return

        # Resize image if too big
        ih, iw = image.shape[:2]
        scale = min(self.cell_width / iw, self.cell_height / ih, 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(iw * scale), int(ih * scale)))

        # Paste the image into the correct slot
        x0 = x * self.cell_width
        y0 = y * self.cell_height

        h, w = image.shape[:2]

        self.display[y0 : y0 + h, x0 : x0 + w] = image

    def show(self, window_name="Display"):
        """Shows the display"""
        cv2.imshow(window_name, self.display)

    def clear(self):
        """Clears the display"""
        self.display.fill(0)
