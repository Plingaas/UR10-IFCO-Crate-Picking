import rtde_control

class Controller:

    def __init__(self, ip):
        try:
            self.controller = rtde_control.RTDEControlInterface(ip)
        except Exception as e:
            print(f"Failed to connect to robot control interface with exception {e}.")
