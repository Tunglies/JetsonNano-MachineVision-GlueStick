import time
import Jetson.GPIO as GPIO


class Status:
    def __init__(self, pin: int = 12) -> None:
        self.status = False

        self.pin = pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT)
    
    def __del__(self):
        self.reset()
    
    def reset(self):
        self.status = False
        GPIO.cleanup()
    
    def __beep(self):
        while True:
            if not self.status:
                break
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(self.pin, GPIO.LOW)
            time.sleep(0.5)
            
    def beep(self):
        self.__beep()

    def set_status(self, status: bool):
        self.status = status