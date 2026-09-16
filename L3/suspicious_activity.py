def track_and_analyze_behavior(person_id, duration_seconds, config_threshold):
    if duration_seconds >= config_threshold:
        return True, "SUSPICIOUS: Stationary for too long"
    return False, "NORMAL"
