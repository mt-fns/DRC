# module for PID control formula

class PidController :
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0



    def control_angle(self, angle):
        """ takes angle and outputs angle to be turned to

        Args:
            angle (angle): input angle
        output:
            angle (angle): output angle
        """
        if angle < 4 and angle > -4:
            error = 0
            self.previous_angle = error
            return angle
        
        error = angle
        proportional = error
        derivative = error - self.previous_error

        output_angle = self.kp * proportional + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output_angle

    