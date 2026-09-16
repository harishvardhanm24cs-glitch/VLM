import math

class MovementAnalyzer:
    def __init__(self, movement_threshold_pixels=20):
        self.threshold = movement_threshold_pixels
        
    def calculate_displacement(self, p1, p2):
        return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
        
    def analyze_movement(self, track_data):
        history = track_data.get("movement_history", [])
        if len(history) < 2:
            return "stationary", 0.0, 0.0
            
        first_pos = history[0]
        last_pos = history[-1]
        
        displacement = self.calculate_displacement(first_pos, last_pos)
        state = "moving" if displacement > self.threshold else "stationary"
        
        stationary_duration = 0.0
        for i in range(len(history)-1, 0, -1):
            disp = self.calculate_displacement(history[i-1], history[-1])
            if disp <= self.threshold:
                stationary_duration = history[-1]['time'] - history[i-1]['time']
            else:
                break
                
        return state, displacement, stationary_duration
