import RPi.GPIO as GPIO
from time import sleep



pin = 11				# PWM pin connected to LED
GPIO.setwarnings(False)			#disable warnings
GPIO.setmode(GPIO.BOARD)		#set pin numbering system .BOARD is chip pin names, .BCM is GPIO numbers
GPIO.setup(pin,GPIO.OUT)
pi_pwm = GPIO.PWM(pin,50)		#create PWM instance with frequency (pin, frequency)
pi_pwm.start(0)				#start PWM of required Duty Cycle 
while(True):
    for duty in range (5,10):
        pi_pwm.ChangeDutyCycle(duty)
        sleep(1)

def set_angle(angle):
    duty = (100/270) * angle
    pi_pwm.ChangeDutyCycle(duty)
