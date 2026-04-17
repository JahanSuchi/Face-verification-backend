## AI Face Verification & Liveness Detection System

A production-style FastAPI-based biometric verification API that extracts faces from passport images, verifies identity using deep face recognition models, and performs liveness checks using head pose and facial expression analysis.

## Key Features
- Passport image upload and automatic face extraction
- Multi-strategy face detection pipeline (MediaPipe + Haar Cascade fallback)
- Face verification using DeepFace (ArcFace embedding model)
- Robust matching with multiple detector backends (retinaface, mtcnn, opencv, etc.)
- Head pose estimation (look left / right / forward)
- Facial emotion & liveness detection using MediaPipe blendshapes
- Image preprocessing (lighting normalization, resizing, cropping)
- Temporary file handling and cleanup for production safety
- RESTful API built using FastAPI with CORS support

## Tech Stack
- Backend: FastAPI
- Computer Vision: OpenCV, MediaPipe
- Face Recognition: DeepFace (ArcFace)
- Models: Face Mesh, Face Detection, Face Landmarker
- Math / ML: NumPy
- Image Processing: PIL, OpenCV
- Runtime: Uvicorn
