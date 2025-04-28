import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from core.ft300_reader import FT300Reader

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