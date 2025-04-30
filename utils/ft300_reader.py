import struct
import threading
import time
import numpy as np
import serial

class FT300Reader(threading.Thread):
    def __init__(self) -> None:
        super().__init__()
        self.daemon = True
        self.running = True
        self.ser = serial.Serial("COM5", 19200, timeout=0.01)
        self.new_data_callback = None
        self.start()
    
    def run(self):
        self.start_sensor()
        while self.running:
            has_data = self.is_data_available()
            if not has_data:
                time.sleep(0.001)
                continue

            data = self.read_data()
            if data is None:
                continue

            wrench = self.unpack(data)
            if self.new_data_callback:
                self.new_data_callback(wrench)


    def start_sensor(self):
        cmd = b"\x09\x10\x01\x9a\x00\x01\x02\x02\x00"
        self.ser.write(cmd + self.compute_crc(cmd))

    def is_data_available(self):
        return self.ser.in_waiting > 0

    def read_data(self):
        valid_header = self.verify_header()
        if not valid_header:
            return None

        packet = self.read_body()
        if not packet:
            return None
        return packet

    def verify_header(self):
        is_header = self.ser.read(1) == b"\x20" and self.ser.read(1) == b"\x4e"
        return is_header

    def read_body(self):
        packet = self.ser.read(12)
        return packet if len(packet) == 12 else None

    def unpack(self, packet):
        raw = np.array(struct.unpack("<hhhhhh", packet))
        f_raw = raw[:3] * 0.01 # Scale to N
        t_raw = raw[3:6] * 0.001 # Scale to Nm
        return np.concatenate([f_raw, t_raw]).reshape((6,1))

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

    def shutdown(self):
        self.running = False
        self.ser.close()
