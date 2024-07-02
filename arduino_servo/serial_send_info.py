import serial
import time

# Set up the serial connection
ser = serial.Serial('/dev/serial0', 9600)  # Adjust the port and baud rate as needed

try:
    while True:
        ser.write(b'Hello, Arduino!\n')  # Send data to the Arduino
        time.sleep(1)  # Wait for a second
except KeyboardInterrupt:
    ser.close()  # Close the serial connection when exiting the script