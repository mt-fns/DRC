import RPi.GPIO as GPIO
from time import sleep
from gpiozero import AngularServo

pwm2_pin = 29
dir2_pin = 31
pwm1_pin = 33
dir1_pin = 35
servo_pin = 17


def setup_pi():
    #motor pins are pin names, servo is GPIO name
    GPIO.setwarnings(False)			#disable warnings
    GPIO.setmode(GPIO.BOARD)		#set pin numbering system
    GPIO.setup(pwm2_pin,GPIO.OUT)
    GPIO.setup(dir2_pin, GPIO.OUT)
    GPIO.setup(pwm1_pin,GPIO.OUT)
    GPIO.setup(dir1_pin, GPIO.OUT)
    pwm2 = GPIO.PWM(pwm2_pin,50)		#create PWM instance with frequency (pin, frequency)
    pwm2.start(0)				#start PWM of required Duty Cycle
    pwm1 = GPIO.PWM(pwm1_pin,50)
    pwm1.start(0)
    GPIO.output(dir2_pin, GPIO.HIGH)
    GPIO.output(dir2_pin, GPIO.LOW)
    #check these directions

    return servo_pin

def drive_servo(angle):
    servo = AngularServo(servo_pin, min_pulse_width=0.0006, max_pulse_width=0.0023)
    servo.angle = angle