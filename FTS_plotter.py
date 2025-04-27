import sys
import struct
import threading
import time
from array import array
from queue import Queue
import serial
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


class FT300Reader(threading.Thread):
    def __init__(self) -> None:
        super().__init__()
        self.daemon = True
        self.running = True
        self.ser = serial.Serial("COM5", 19200)
        self.wrench = array("f", [0.0] * 6)
        self.queue = Queue()
        self.new_data = False
        self.start()

    def run(self):
        cmd = b"\x09\x10\x01\x9a\x00\x01\x02\x02\x00"
        self.ser.write(cmd + self.compute_crc(cmd))
        while self.running:
            if self.ser.in_waiting == 0:
                time.sleep(0.001)
                continue
            if self.ser.read(1) == b"\x20" and self.ser.read(1) == b"\x4e":
                packet = self.ser.read(12)
                if len(packet) == 12:
                    raw = struct.unpack("<hhhhhh", packet)
                    self.wrench[0] = raw[0] / 100.0
                    self.wrench[1] = raw[1] / 100.0
                    self.wrench[2] = raw[2] / 100.0
                    self.wrench[3] = raw[3] / 1000.0
                    self.wrench[4] = raw[4] / 1000.0
                    self.wrench[5] = raw[5] / 1000.0
                    self.queue.put(self.wrench[:])
                    self.new_data = True

    def compute_crc(self, data):
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return struct.pack("<H", crc)

    def get_wrench(self):
        if self.new_data:
            self.new_data = False
            return self.queue.get()
        return None

    def shutdown(self):
        self.running = False
        self.ser.close()


class ForceTorquePlotter(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.init_ui()
        self.reader = FT300Reader()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(1)  # Update every 30 ms

    def init_ui(self):
        self.setWindowTitle("Force & Torque Live Plotter")
        self.setStyleSheet("background-color: #121212; color: white;")
        layout = QtWidgets.QGridLayout()

        labels = ["Force X", "Force Y", "Force Z", "Torque X", "Torque Y", "Torque Z"]
        self.plots = []
        self.curves = []
        self.data = [np.zeros(500) for _ in range(6)]

        neon_colors = ["#00FFFF", "#39FF14", "#FF1493", "#FFD700", "#FF4500", "#7DF9FF"]

        for i, label in enumerate(labels):
            plot = pg.PlotWidget()
            plot.setBackground("#121212")
            plot.showGrid(x=True, y=True, alpha=0.3)
            if i < 3:
                plot.setYRange(-100, 100)
            else:
                plot.setYRange(-10, 10)

            plot.setTitle(label, color="w", size="12pt")
            curve = plot.plot(pen=pg.mkPen(neon_colors[i], width=2))

            self.plots.append(plot)
            self.curves.append(curve)
            layout.addWidget(plot, i // 3, i % 3)

        self.setLayout(layout)

    def update_plots(self):
        # Always take the latest wrench data, discard old
        latest_wrench = None
        while not self.reader.queue.empty():
            latest_wrench = self.reader.queue.get()

        if latest_wrench:
            for i in range(6):
                self.data[i][:-1] = self.data[i][1:]
                self.data[i][-1] = latest_wrench[i]
                self.curves[i].setData(self.data[i])

    def closeEvent(self, event):
        self.reader.shutdown()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = ForceTorquePlotter()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())
