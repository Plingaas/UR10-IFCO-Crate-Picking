import numpy as np
import threading
import time
from utils.ft300_reader import FT300Reader
from utils.low_pass_filter import LowPassFilter
from concurrent.futures import Future
from utils.helper import print_with_time

class FT300:

    def __init__(self) -> None:

        self.reader = FT300Reader()
        self.reader.new_data_callback = self.update_data

        self.wrench = np.zeros((6, 1))
        self.filter = LowPassFilter(100.0, 5.0, values=6)
        self.lock = threading.Lock()

        self.offsets = np.zeros((6, 1))

        self.calibrating = False
        self.future_cal = None
        self.n_cal = 500
        self.measurements = np.zeros((self.n_cal,6,1))
        self.cal_index = 0

        self.new_data = False

    def update_data(self, data):
        with self.lock:
            if not self.calibrating:
                self.wrench = self.filter.filter(data) - self.offsets
                self.new_data = True
                return
        
            # Calibrating
            if self.cal_index < self.n_cal:
                self.measurements[self.cal_index] = data
                self.cal_index += 1
            else:
                self.offsets = np.mean(self.measurements, axis = 0)
                self.calibrating = False
                self.future_cal.set_result(True)

    def calibrate(self):
        self.future_cal = Future()
        print_with_time("FT300", "Calibrating...")
        with self.lock:
            self.calibrating = True
        self.future_cal.result()
        print_with_time("FT300", f"Finished calibrating (\n\
                                    f_x: {self.offsets[0][0]}, \n\
                                    f_y: {self.offsets[1][0]}, \n\
                                    f_z: {self.offsets[2][0]}, \n\
                                    t_x: {self.offsets[3][0]}, \n\
                                    t_y: {self.offsets[4][0]}, \n\
                                    t_z: {self.offsets[5][0]}, \n\
                                )")

    def get_wrench(self):
        with self.lock:
            self.new_data = False
            return self.wrench.flatten()

    def get_unread_wrench(self):
        while True:
            with self.lock:
                if self.new_data:
                    self.new_data = False
                    return self.wrench
            time.sleep(0.001)
    
    def estimate_weight(self, n=10):
        print_with_time("FT300", "Estimating item weight...")
        est_weight = 0
        i = 0
        while i < n:
            got_data = False
            with self.lock:
                if self.new_data:
                    self.new_data = False
                    est_weight += self.wrench.flatten()[2]
                    i += 1
            if not got_data:
                time.sleep(0.001)

        est_weight = est_weight / 9.81 / n
        print_with_time("FT300", f"Estimated item weight: {round(est_weight, 3)} kg.")
        return est_weight