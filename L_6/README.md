# L_6 Face Recognition Module

This module handles Face Recognition for the VLM Surveillance platform.

**CRITICAL RULE:**
This is an independent frontend/backend additional module.
It operates without disrupting YOLO, ByteTrack, L1/L2, L3, L5, ANPR, VLM, IoT, Blockchain, or the existing camera pipeline.

## Status
Foundation initialized. Schemas and testing infrastructure are in place.

## Components
- `face_detector.py`: Detect faces in crops.
- `face_quality.py`: Ensure faces meet quality thresholds.
- `face_aligner.py`: Align facial crops for embedding.
- `face_embedder.py`: Generate feature vectors.
- `face_matcher.py`: Match vectors against the database.
- `enrollment.py`: Add new identities.
- `database.py`: Store and retrieve vectors.
- `recognition_service.py`: Main module orchestrator.
