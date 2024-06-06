import RPi.GPIO as GPIO
from time import sleep
pwm2_pin = 29
dir2_pin = 31
pwm1_pin = 33
dir1_pin = 35
servo_pin = 11

GPIO.setwarnings(False)			#disable warnings
GPIO.setmode(GPIO.BOARD)		#set pin numbering system
GPIO.setup(pwm2_pin,GPIO.OUT)
GPIO.setup(dir2_pin, GPIO.OUT)
GPIO.setup(pwm1_pin,GPIO.OUT)
GPIO.setup(dir1_pin, GPIO.OUT)
GPIO.setup(servo_pin,GPIO.OUT)
pwm2 = GPIO.PWM(pwm2_pin,50)		#create PWM instance with frequency (pin, frequency)
pwm2.start(0)				#start PWM of required Duty Cycle 
pwm1 = GPIO.PWM(pwm1_pin,50)
pwm1.start(0)
GPIO.output(dir2_pin, GPIO.HIGH)
servo_pwm = GPIO.PWM(servo_pin,50)		#create PWM instance with frequency (pin, frequency)

#Turn wheels straight
servo_pwm.start(6)				#start PWM of required Duty Cycle


#go slow
pwm1.ChangeDutyCycle(5)
pwm2.ChangeDutyCycle(5)

#turn right (or is it left)
servo_pwm.start(10)

#straight again
servo_pwm.start(6)

#go faster
pwm1.ChangeDutyCycle(20)
pwm2.ChangeDutyCycle(20)


while(True):
    duty = input("motor 1\n")
    if(duty == 0):
        break
    pwm2.ChangeDutyCycle(int(duty))
    print("changing ", duty)

    duty = input("motor 2\n")
    if(duty == 0):
        break
    pwm1.ChangeDutyCycle(int(duty))
    print("changing ", duty)

    duty = input("servo\n")
    if(duty == 0):
        break
    servo_pwm.ChangeDutyCycle(int(duty))
    print("changing ", duty)
#for duty in range (0,50, 5):
 #   pwm2.ChangeDutyCycle(duty)
  #  print("changing ", duty)
   # sleep(1)

def set_angle(angle):
    duty = (100/270) * angle
    pwm2.ChangeDutyCycle(duty)