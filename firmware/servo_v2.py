from gpiozero import AngularServo
from time import sleep

# pwm1_pin = 13
# dir1_pin = 19
# pwm2_pin = 5
# dir2_pin = 6
servo_pin = 17
#GPIO names not pin names

servo = AngularServo(servo_pin, min_pulse_width=0.0006, max_pulse_width=0.0023)

while (True):
    servo.angle = 90
    sleep(2)
    servo.angle = 0
    sleep(2)
    servo.angle = -90
    sleep(2)