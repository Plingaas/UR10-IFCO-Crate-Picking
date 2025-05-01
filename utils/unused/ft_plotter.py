import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from utils.ft300_reader import FT300Reader
import threading
from utils.low_pass_filter import LowPassFilter

class ForceTorquePlotter(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.init_ui()
        self.reader = FT300Reader()
        self.reader.new_data_callback = self.update_plots
        self.timer = QtCore.QTimer()
        self.lock = threading.Lock()
        self.filter = LowPassFilter(100.0, 5.0)

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

    def update_plots(self, data):
        with self.lock:
            data = self.filter.filter(data)
            for i, value in enumerate(data):
                self.data[i][:-1] = self.data[i][1:]
                self.data[i][-1] = value
                self.curves[i].setData(self.data[i])

    def closeEvent(self, event):
        self.reader.shutdown()
        event.accept()