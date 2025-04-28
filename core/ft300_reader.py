import struct
import threading
import time
from array import array
from queue import Queue
import serial

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