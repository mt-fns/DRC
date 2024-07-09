# module for PID control formula

class PidController():
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.previous_error = 0
        self.integral = 0



    def control_angle(self, angle):
        """ takes angle and outputs angle to be turned to

        Args:
            angle (angle): input angle
        output:
            angle (angle): output angle
        """

        error = angle
        dt = 1

        proportional = error
        derivative = (error - self.previous_error) / dt
        output_angle = self.kp * proportional + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output_angle
