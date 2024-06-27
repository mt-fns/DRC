from gpiozero import PhaseEnableMotor
from gpiozero import AngularServo
from time import sleep

#GPIO names not board
pwm2_pin = 5#29
dir2_pin = 6#31
pwm1_pin = 13#33
dir1_pin = 19#35
servo_pin = 11#17

motor1 = PhaseEnableMotor(dir1_pin, pwm1_pin)
motor2 = PhaseEnableMotor(dir2_pin, pwm2_pin)

servo = AngularServo(servo_pin, min_pulse_width=0.0006, max_pulse_width=0.0023)

#Turn wheels straight
servo.angle = 0

#go slow
motor1.forward(0.1)
motor2.backwards(0.1)

sleep(2)
#go faster
motor1.forward(0.25)
motor2.backwards(0.25)
sleep(2)
#turn right (or is it left)
servo.angle = 10
sleep(2)
#straight again
servo.angle = 0
sleep(2)
#go faster
motor1.forward(0.4)
motor2.backwards(0.4)
