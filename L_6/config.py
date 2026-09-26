import os


class Config:

    FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.75"))

    FACE_UNCERTAIN_MARGIN = float(os.getenv("FACE_UNCERTAIN_MARGIN", "0.05"))


config = Config()
