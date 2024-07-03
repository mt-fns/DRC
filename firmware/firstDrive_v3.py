from gpiozero import PhaseEnableMotor
from gpiozero import AngularServo
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory

#GPIO names not board
pwm2_pin = 5#29
dir2_pin = 6#31
pwm1_pin = 13#33
dir1_pin = 19#35
servo_pin = 17#17

factory = PiGPIOFactory()

motor1 = PhaseEnableMotor(dir1_pin, pwm1_pin)
motor2 = PhaseEnableMotor(dir2_pin, pwm2_pin)

servo = AngularServo(servo_pin, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
servo.angle = 2
angle = 2
speed = 0
while(True):
    data = input("input:")
    if (data == "a"):
        angle = min(15, angle - 4)
        servo.angle = angle
    if (data == "d"):
        angle = max(15, angle + 4)
        servo.angle = angle
    if (data == "w"):
        speed = speed + 0.1
        if (speed < 0):
            motor1.forward(-speed)
            motor2.backward(-speed)
        if (speed > 0):
            speed = min(0.6, speed)
            motor1.backward(speed)
            motor2.forward(speed)
    if (data == "s"):
        speed = speed - 0.1
        if (speed < 0):
            speed = max(-0.6, speed)
            motor1.forward(-speed)
            motor2.backward(-speed)
        if (speed > 0):
            motor1.backward(speed)
            motor2.forward(speed)
#Turn wheels straight
servo.angle = 2

#go slow
motor1.backward(0.1)
motor2.forward(0.1)

sleep(1)
#go faster
servo.angle = -18
motor1.backward(0.4)
motor2.forward(0.4)
sleep(10)
#turn right (or is it left)
servo.angle = -20
sleep(8)
#straight again
servo.angle = 0
sleep(2)
#go faster
motor1.backward(0.4)
motor2.forward(0.4)
