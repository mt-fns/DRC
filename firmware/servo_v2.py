from gpiozero import AngularServo
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory
import pigpio
# PIGPIO_ADDR=soft
# PIGPIO_PORT=8888

# pwm1_pin = 13
# dir1_pin = 19
# pwm2_pin = 5
# dir2_pin = 6
servo_pin = 17
#GPIO names not pin names
factory = PiGPIOFactory()
pi = pigpio.pi('soft', 8888)
servo = AngularServo(servo_pin, min_pulse_width=0.0005, max_pulse_width=0.00255, pin_factory=factory)

while (True):
    data = input("angle")
    servo.angle = float(data)