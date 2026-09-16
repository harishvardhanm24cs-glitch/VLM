import unittest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from L3.movement import MovementAnalyzer
from L3.suspicious_activity import SuspiciousActivityRules
from L3.event_engine import EventEngine

class TestBehaviourEngine(unittest.TestCase):
    
    def test_movement_stationary(self):
        analyzer = MovementAnalyzer(movement_threshold_pixels=20)
        track_data = {
            "movement_history": [
                {"x": 100, "y": 100, "time": 1.0},
                {"x": 105, "y": 105, "time": 2.0},
                {"x": 102, "y": 103, "time": 3.0}
            ]
        }
        state, disp, dur = analyzer.analyze_movement(track_data)
        self.assertEqual(state, "stationary")
        self.assertEqual(dur, 2.0) # 3.0 - 1.0
        
    def test_movement_moving(self):
        analyzer = MovementAnalyzer(movement_threshold_pixels=20)
        track_data = {
            "movement_history": [
                {"x": 100, "y": 100, "time": 1.0},
                {"x": 150, "y": 150, "time": 2.0}
            ]
        }
        state, disp, dur = analyzer.analyze_movement(track_data)
        self.assertEqual(state, "moving")
        self.assertEqual(dur, 0.0)
        
    def test_threshold_crossing(self):
        config = {"person_stationary_time": 10}
        rules = SuspiciousActivityRules(config)
        enhanced_track = {
            "class": "person",
            "movement_state": "stationary",
            "stationary_duration": 12.0
        }
        matches = rules.evaluate(enhanced_track)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["event_type"], "stationary_person")
        self.assertEqual(matches[0]["duration"], 12.0)
        
    def test_duplicate_alert_prevention(self):
        engine = EventEngine("outputs/alerts")
        track = {"track_id": 7, "class": "person", "current_bbox": [0,0,10,10]}
        matches = [{"event_type": "stationary_person", "severity": "high", "duration": 12.0}]
        
        # First process
        engine.process_rule_matches(track, matches, 100.0)
        self.assertEqual(len(engine.all_events), 1)
        
        # Second process with same event (simulate next frame)
        engine.process_rule_matches(track, matches, 101.0)
        self.assertEqual(len(engine.all_events), 1) # Should not duplicate
        
if __name__ == '__main__':
    unittest.main()
