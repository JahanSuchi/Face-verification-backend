from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace 
import shutil 
import os 
import time 
import tempfile
from PIL import Image
from dotenv import load_dotenv
from mediapipe.tasks import python

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.4
)

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize the face landmarker (NEW API)
base_options = python.BaseOptions(
    model_asset_buffer=open('face_landmarker.task', "rb").read()
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Folder setup
UPLOAD_FOLDER = "./uploads"
EXTRACTED_FOLDER = "./extracted_faces"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_FOLDER, exist_ok=True)


def normalize_lighting(image_path: str):
    """Normalize brightness and contrast using CLAHE"""
    image = cv2.imread(image_path)
    if image is None:
        return
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    normalized = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)
    cv2.imwrite(image_path, normalized)
    print(f"✅ Lighting normalized: {image_path}")


def extract_face(image_path: str, face_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    print(f"📸 Processing image: {image_path}")
    print(f"📏 Original image size: {image.shape[1]}x{image.shape[0]}")
    
    h, w = image.shape[:2]
    
    # Preprocess: Resize small images
    if min(h, w) < 400:
        scale = 600 / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"⬆️ Upscaled image to: {new_w}x{new_h}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detection_found = False
    results = None
    
    # Strategy 1: Standard detection (confidence 0.4)
    print("🔍 Strategy 1: Standard detection...")
    results = face_detection.process(image_rgb)
    if results.detections:
        detection_found = True
        print(f"✅ Found face with standard detection (confidence: {results.detections[0].score[0]:.2f})")
    
    # Strategy 2: Lower confidence threshold (0.2)
    if not detection_found:
        print("🔍 Strategy 2: Lower confidence threshold...")
        face_detection_low = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.2
        )
        results = face_detection_low.process(image_rgb)
        if results.detections:
            detection_found = True
            print(f"✅ Found face with low confidence (confidence: {results.detections[0].score[0]:.2f})")
    
    # Strategy 3: Close-range model
    if not detection_found:
        print("🔍 Strategy 3: Close-range model...")
        face_detection_close = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.3
        )
        results = face_detection_close.process(image_rgb)
        if results.detections:
            detection_found = True
            print(f"✅ Found face with close-range model")
    
    # Strategy 4: Enhanced contrast (CLAHE)
    if not detection_found:
        print("🔍 Strategy 4: Enhanced contrast...")
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = face_detection.process(enhanced_rgb)
        if results.detections:
            detection_found = True
            image = enhanced
            print(f"✅ Found face with enhanced contrast")
    
    # Strategy 5: Haar Cascade fallback
    if not detection_found:
        print("🔍 Strategy 5: Haar Cascade fallback...")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        if len(faces) > 0:
            detection_found = True
            print(f"✅ Haar Cascade detected {len(faces)} face(s)")
            
            x, y, width, height = max(faces, key=lambda rect: rect[2] * rect[3])
            
            padding = 0.15
            x_pad = int(width * padding)
            y_pad = int(height * padding)
            
            h, w = image.shape[:2]
            x = max(0, x - x_pad)
            y = max(0, y - y_pad)
            width = min(w - x, width + 2 * x_pad)
            height = min(h - y, height + 2 * y_pad)
            
            face_image = image[y:y + height, x:x + width]
            if face_image.size > 0:
           
                output_path = os.path.join(EXTRACTED_FOLDER, "face.png")
                cv2.imwrite(output_path, face_image)
                print(f"✅ Face extracted using Haar Cascade: {output_path}")
                return output_path
    
    # Process MediaPipe detections
    if detection_found and results and results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        width = int(bboxC.width * w)
        height = int(bboxC.height * h)
        
        padding = 0.15
        x_pad = int(width * padding)
        y_pad = int(height * padding)
        
        x = max(0, x - x_pad)
        y = max(0, y - y_pad)
        width = min(w - x, width + 2 * x_pad)
        height = min(h - y, height + 2 * y_pad)
        
        face_image = image[y:y + height, x:x + width]
        
        if face_image.size == 0:
            print("❌ Face extraction failed - empty region")
            return False
        
        print(f"📏 Extracted face size: {width}x{height}")
        
        output_path = os.path.join(EXTRACTED_FOLDER, "face.png")
        cv2.imwrite(output_path, face_image)
        
        print(f"✅ Face saved to: {output_path}")
        return output_path
    
    print("❌ No face detected after all strategies")
    return False


@app.post("/upload_passport/")
async def upload_passport(file: UploadFile = File(...)):
    """Upload a passport image, detect the face, and extract it."""
    passport_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(passport_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    face_path = os.path.join(EXTRACTED_FOLDER, f"face_{file.filename}")
    extracted_face = extract_face(passport_path, face_path)

    if extracted_face:
        return JSONResponse(
            {"message": "Passport face extracted successfully", "face_path": extracted_face}
        )
    else:
        raise HTTPException(status_code=400, detail="No face detected in passport image")
    

def crop_face_only(image_path: str) -> str:
    """Crop only the face region (eyes, nose, mouth) excluding hijab/hair/background"""
    image = cv2.imread(image_path)
    if image is None:
        return image_path

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_face_mesh_static = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3
    )
    
    results = mp_face_mesh_static.process(image_rgb)
    
    if results.multi_face_landmarks:
        h, w = image.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        FACE_OVAL_INDICES = [
            10, 338, 297, 332, 284, 251,
            389, 356, 454, 323, 361,
            288, 397, 365, 379, 378,
            400, 377, 152,
            148, 176, 149, 150,
            136, 172, 58, 132,
            93, 234, 127, 162,
            21, 54, 103, 67, 109
        ]
        
        points = np.array([
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in FACE_OVAL_INDICES
        ])
        
        x, y, bw, bh = cv2.boundingRect(points)
        
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.05)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        bw = min(w - x, bw + 2 * pad_x)
        bh = min(h - y, bh + 2 * pad_y)
        
        face_only = image[y:y+bh, x:x+bw]
        
        if face_only.size > 0:
            # Save over original for matching
            cv2.imwrite(image_path, face_only)
            
            os.makedirs("./face_crops", exist_ok=True)
            filename = os.path.splitext(os.path.basename(image_path))[0]
            crop_save_path = f"./face_crops/{filename}_faceonly.png"
            cv2.imwrite(crop_save_path, face_only)
            print(f"✅ Face-only crop saved: {crop_save_path}")
 
            
            return image_path
        
    print("⚠️ Face oval crop failed, using full image")
    return image_path


def compareFace(realfilepath: str) -> bool:
    passport_photo = os.listdir(EXTRACTED_FOLDER)
    if not passport_photo:
        return False

    passport_extracted_face_path = os.path.join(EXTRACTED_FOLDER, passport_photo[0])
    #normalize_lighting(passport_extracted_face_path)
    normalize_lighting(realfilepath)
    #crop_face_only(passport_extracted_face_path)
    crop_face_only(realfilepath)

    backends = ['retinaface', 'mtcnn', 'ssd', 'opencv']

    for backend in backends:
        try:
            result = DeepFace.verify(
                img1_path=passport_extracted_face_path,
                img2_path=realfilepath,
                model_name="ArcFace",
                distance_metric="cosine",
                enforce_detection=False,
                detector_backend=backend,
                align=True
            )

            distance = result.get("distance", 1.0)
            THRESHOLD = 0.72
            is_match = distance < THRESHOLD

            print(f"✅ Backend: {backend} | ArcFace Distance: {distance:.4f} | Match: {is_match}")
            return is_match

        except Exception as e:
            print(f"⚠️ Backend '{backend}' failed: {e}, trying next...")
            continue

    return False

@app.post("/match_face/")
async def match_face(file: UploadFile = File(...)):
    """Match uploaded face with passport photo"""
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Read image
        image = cv2.imread(temp_file_path)

    
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Flip horizontally (mirror correction)
        image = cv2.flip(image, 1)
        cv2.imwrite(temp_file_path, image)

        print(f"📸 Reading face image: {temp_file_path}")

        # Resize if too large
        height, width = image.shape[:2]
        max_dimension = 1024
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            cv2.imwrite(temp_file_path, image)
            print(f"📏 Resized image to {new_width}x{new_height}")

        print(f"🔍 Attempting face match...")
        if compareFace(temp_file_path):
            return {"result": True, "message": "Face matched successfully"}
        else:
            return {"result": False, "message": "Face did not match"}

    except Exception as e:
        print(f"❌ Exception in match_face: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if temp_file_path:
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    print(f"🗑️ Cleaned up temp file: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Could not delete temp file: {cleanup_error}")

        try:
            crops_folder = "./face_crops"
            if os.path.exists(crops_folder):
                for f in os.listdir(crops_folder):
                    os.remove(os.path.join(crops_folder, f))
                print("🗑️ Face crop files cleaned up")
        except Exception as e:
            print(f"⚠️ Could not clean face_crops: {e}")



def analyze_head_pose(image: np.ndarray, command: str) -> bool:
    """Analyze head pose direction (forward, left, right)"""
    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []

    results = face_mesh.process(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1]
            ])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x_angle = angles[0] * 360
            y_angle = angles[1] * 360

            print(f"📐 x_angle: {x_angle:.2f} | y_angle: {y_angle:.2f}")

            if command.lower() == "look forward":
                return -10 <= y_angle <= 10 and -10 <= x_angle <= 20
            elif command.lower() == "look left":
                return y_angle < -10
            elif command.lower() == "look right":
                return y_angle > 10

    return False


def detect_emotion(face_blendshapes, emotion: str) -> bool:
    """Detect facial expressions (smile, eye direction)"""
    print(emotion)
    
    if emotion == "smile":
        parameters = {
            "mouthSmileLeft": None,
            "mouthSmileRight": None
        }
        for blendshape in face_blendshapes:
            if blendshape.category_name in parameters:
                parameters[blendshape.category_name] = blendshape.score

        if parameters["mouthSmileLeft"] >= 0.8 and parameters["mouthSmileRight"] >= 0.8:
            return True
        else:
            return False

    elif emotion == "head still, eyes left":
        parameters = {
            "eyeLookOutLeft": None,
            "eyeLookInRight": None,
            "mouthSmileLeft": None,
            "mouthSmileRight": None,
            "jawOpen": None
        }
        for blendshape in face_blendshapes:
            if blendshape.category_name in parameters:
                parameters[blendshape.category_name] = blendshape.score

        if parameters["mouthSmileLeft"] >= 0.6 and parameters["mouthSmileRight"] >= 0.6:
            return False
        elif parameters["jawOpen"] >= 0.6:
            return False
        else:
            if parameters["eyeLookOutLeft"] >= 0.7 and parameters["eyeLookInRight"] >= 0.7:
                return True
            else:
                return False

    elif emotion == "head still, eyes right":
        parameters = {
            "eyeLookOutRight": None,
            "eyeLookInLeft": None,
            "mouthSmileLeft": None,
            "mouthSmileRight": None,
            "jawOpen": None
        }
        for blendshape in face_blendshapes:
            if blendshape.category_name in parameters:
                parameters[blendshape.category_name] = blendshape.score

        if parameters["eyeLookOutRight"] >= 0.7 and parameters["eyeLookInLeft"] >= 0.7:
            return True
        else:
            return False

    return False


@app.post("/detect-emotion/")
async def detect_emotion_api(file: UploadFile, emotion: str = Form(...)):
    """Detect emotion or head pose based on uploaded image"""
    
    if emotion.startswith("Look"):
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name

            image = cv2.imread(temp_file_path)
            print(f"📸 Head pose image: {temp_file_path}")
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            # Note: realfilepath param removed from analyze_head_pose
            result = analyze_head_pose(image, emotion)
            print({"emotion": emotion, "result": result})
            return JSONResponse(content={"emotion": emotion, "result": result})

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
        finally:
            try:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except:
                pass
    else:
        try:
            image_data = await file.read()
            np_image = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            detection_result = detector.detect(mp_image)
            if not detection_result.face_blendshapes:
                return JSONResponse(content={"error": "No face detected."}, status_code=400)

            face_blendshapes = detection_result.face_blendshapes[0]
            result = detect_emotion(face_blendshapes, emotion)
            print({"emotion": emotion, "result": result})
            return {"emotion": emotion, "result": result}

        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)