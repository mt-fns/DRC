import RPi.GPIO as GPIO
from time import sleep
pwm2_pin = 35
dir2_pin = 37
pin = 11				# PWM pin connected to LED
GPIO.setwarnings(False)			#disable warnings
GPIO.setmode(GPIO.BOARD)		#set pin numbering system
GPIO.setup(pwm2_pin,GPIO.OUT)
GPIO.setup(dir2_pin, GPIO.OUT)
pwm2 = GPIO.PWM(pwm2_pin,50)		#create PWM instance with frequency (pin, frequency)
pwm2.start(0)				#start PWM of required Duty Cycle 
GPIO.output(dir2_pin, GPIO.HIGH)
#while(True):
for duty in range (5,10):
    pwm2.ChangeDutyCycle(0)
    sleep(1)

def set_angle(angle):
    duty = (100/270) * angle
    pwm2.ChangeDutyCycle(duty)
