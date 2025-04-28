import sys
from PyQt5 import QtWidgets
import pyqtgraph as pg
from utils.ft_plotter import ForceTorquePlotter

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = ForceTorquePlotter()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())
