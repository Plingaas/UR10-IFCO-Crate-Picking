# This project is licensed under the GNU General Public License v3.0 (GPL-3.0) with a Non-Commercial Use Exception.  
# You may use, modify, and share the code for non-commercial purposes only.
#
# See the full license in [LICENSE.md](LICENSE.md) for more details.

import time
from utils.helper import print_with_time
from core.main_controller import MainController

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
        print("Interrupted by user, shutting down...")
        main.shutdown()
