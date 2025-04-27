import time
from utils.helper import print_with_time
from core.MainController import MainController

if __name__ == "__main__":
    main = MainController()

    try:
        main.setup()
        ready = input("Press enter to start picking.")

        main.loop()
        while True:
            time.sleep(1)
            if main.finished and not main.robot.active:
                print_with_time("Main", "Finished order.")
                print("Main", "Shutting down...")
                main.shutdown()
                exit()

    except KeyboardInterrupt:
        print("Shutting down...")
        main.shutdown()
