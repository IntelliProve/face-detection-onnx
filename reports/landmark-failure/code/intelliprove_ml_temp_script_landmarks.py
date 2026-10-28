"""Compare landmark detection on one frame: original vs grayscale (see face-detection-onnx/reports/landmark-failure)."""

from pathlib import Path

import cv2
import numpy as np
from fdlite import face_detection_to_roi
from fdlite.face_landmark import DETECTION_THRESHOLD
from fdlite.transform import image_to_tensor, sigmoid

from intelliprove_ml import NoFaceDetectedError, detect_landmarks, init_detector

FRAME_PATH = Path(r'C:\Users\maart\Downloads\pj1__e96caf44cd6f4c8c97d6423f4dc2d1ee__s1\000000.jpg')
OUT_PATH = FRAME_PATH.with_name(f'{FRAME_PATH.stem}_landmarks_original_vs_grayscale.jpg')


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Drop all colour but keep 3 channels, so the models see the same contrast without the cast."""
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def presence_score(rgb: np.ndarray, roi, detector) -> float:
    """Landmark model face-presence score; FaceLandmark returns [] at or below DETECTION_THRESHOLD."""
    landmark = detector.landmark
    height, width = landmark.input_shape[1:3]
    tensor = image_to_tensor(rgb, roi, output_size=(width, height), keep_aspect_ratio=False, output_range=(0.0, 1.0))
    outputs = landmark.session.run(None, {landmark.input_name: tensor.tensor_data[np.newaxis]})
    return float(sigmoid(outputs[1]).flatten()[-1])


def analyse(name: str, frame: np.ndarray, original: np.ndarray, detector) -> np.ndarray:
    """Print detection/presence/landmark results for one variant and return the annotated original frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    canvas = original.copy()

    detections = detector.detection(rgb)
    print(f'\n[{name}] face detections: {len(detections)}')
    for i, detection in enumerate(detections):
        roi = face_detection_to_roi(detection, (w, h))
        score = presence_score(rgb, roi, detector)
        print(f'  #{i} detection score {detection.score:.3f} | presence {score:.6f} (threshold {DETECTION_THRESHOLD})')
        polygon = np.array(roi.scaled((w, h)).points(), dtype=np.int32)
        cv2.polylines(canvas, [polygon], isClosed=True, color=(255, 128, 0), thickness=2)

    # The production path: what the engine actually gets for this frame
    try:
        face = detect_landmarks(frame, detector)
        print(f'  detect_landmarks: {len(face.coord_list)} landmarks')
        for p in face.coord_list:
            # Same geometry as the input, so the landmarks map 1:1 onto the original frame
            cv2.circle(canvas, (int(p.x), int(p.y)), 1, (0, 255, 0), -1)
        verdict, colour = f'{name}: {len(face.coord_list)} landmarks', (0, 200, 0)
    except NoFaceDetectedError:
        print('  detect_landmarks: NoFaceDetectedError')
        verdict, colour = f'{name}: no landmarks', (0, 0, 255)

    cv2.putText(canvas, verdict, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
    return canvas


if __name__ == '__main__':
    original = cv2.imread(str(FRAME_PATH))
    if original is None:
        raise FileNotFoundError(FRAME_PATH)

    detector = init_detector()
    panels = [analyse(name, frame, original, detector) for name, frame in [('original', original), ('grayscale', to_grayscale(original))]]

    cv2.imwrite(str(OUT_PATH), np.hstack(panels))
    print(f'\nSaved side-by-side to {OUT_PATH}')
