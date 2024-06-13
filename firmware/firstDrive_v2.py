import RPi.GPIO as GPIO
from time import sleep
from gpiozero import AngularServo

#motor pins are pin names, servo is GPIO name
pwm2_pin = 29
dir2_pin = 31
pwm1_pin = 33
dir1_pin = 35
servo_pin = 17

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



servo = AngularServo(servo_pin, min_pulse_width=0.0006, max_pulse_width=0.0023)

#Turn wheels straight
servo.angle = 0

#go slow
pwm1.ChangeDutyCycle(5)
pwm2.ChangeDutyCycle(5)

sleep(2)
#go faster
pwm1.ChangeDutyCycle(20)
pwm2.ChangeDutyCycle(20)
sleep(2)
#turn right (or is it left)
servo.angle = 45
sleep(2)
#straight again
servo.angle = 0
sleep(2)
#go faster
pwm1.ChangeDutyCycle(40)
pwm2.ChangeDutyCycle(40)


# while(True):
#     duty = input("motor 1\n")
#     if(duty == 0):
#         break
#     pwm2.ChangeDutyCycle(int(duty))
#     print("changing ", duty)

#     duty = input("motor 2\n")
#     if(duty == 0):
#         break
#     pwm1.ChangeDutyCycle(int(duty))
#     print("changing ", duty)

#     duty = input("servo\n")
#     if(duty == 0):
#         break
#     servo_pwm.ChangeDutyCycle(int(duty))
#    print("changing ", duty)
#for duty in range (0,50, 5):
 #   pwm2.ChangeDutyCycle(duty)
  #  print("changing ", duty)
   # sleep(1)
