import logging

logger = logging.getLogger(__name__)

class VehicleDB:
    def __init__(self):
        # Mock database of authorized or known vehicles
        self.mock_db = {
            "TN57AB1234": {
                "owner": "Acme Corp",
                "type": "Authorized Delivery",
                "status": "Cleared"
            },
            "XYZ9876": {
                "owner": "John Doe",
                "type": "Staff",
                "status": "Cleared"
            }
        }

    def lookup_plate(self, plate_text):
        """
        Looks up a plate in the database.
        Returns a dict with vehicle info or None if not found.
        """
        # Normalize just in case
        normalized = ''.join(e for e in plate_text if e.isalnum()).upper()
        
        if normalized in self.mock_db:
            return self.mock_db[normalized]
            
        return None
