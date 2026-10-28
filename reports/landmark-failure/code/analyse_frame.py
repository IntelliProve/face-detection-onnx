"""Reproduce every image and number in ../report.md for one frame.

Each variant runs the same steps as intelliprove_ml.detector.detect_landmarks: face detection
(FaceDetection SHORT) -> face_detection_to_roi -> FaceLandmark for every detection, then
'ok' (exactly one face with landmarks), 'no_face' or 'multiple_faces'.

A variant is a pair of images: the one BlazeFace sees and the one the landmark model crops from.
'colour_det_gray_crop' keeps face detection on the colour frame and only grayscales the landmark input.

Usage (from anywhere, uses the fdlite package of this repo):
    python reports/landmark-failure/code/analyse_frame.py [<frame.jpg>]
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fdlite import FaceDetection, FaceDetectionModel, FaceLandmark, face_detection_to_roi  # noqa: E402
from fdlite import face_landmarks_to_render_data  # noqa: E402
from fdlite.face_landmark import DETECTION_THRESHOLD  # noqa: E402
from fdlite.render import Colors, detections_to_render_data, render_to_image  # noqa: E402
from fdlite.transform import image_to_tensor, sigmoid  # noqa: E402

DEFAULT_FRAME = Path(r'C:\Users\maart\Downloads\pj1__e96caf44cd6f4c8c97d6423f4dc2d1ee__s1\000000.jpg')
IMG_DIR = Path(__file__).resolve().parents[1] / 'img'
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263  # face mesh indices, inter-ocular distance (IOD) reference
RED, GREEN = (230, 40, 40), (40, 200, 70)


def grayscale(img: Image.Image) -> Image.Image:
    return img.convert('L').convert('RGB')


def white_balance(img: Image.Image) -> Image.Image:
    """Gray-world: scale each channel so its mean equals the overall mean."""
    a = np.asarray(img).astype(np.float32)
    means = a.reshape(-1, 3).mean(0)
    return Image.fromarray(np.clip(a * (means.mean() / means), 0, 255).astype(np.uint8))


def landmark_input(img: Image.Image, roi, landmark: FaceLandmark) -> np.ndarray:
    """The exact (H, W, 3) float tensor in [0, 1] that FaceLandmark feeds the model."""
    h, w = landmark.input_shape[1:3]
    return image_to_tensor(img, roi, output_size=(w, h), keep_aspect_ratio=False, output_range=(0.0, 1.0)).tensor_data


def presence_logit(tensor: np.ndarray, landmark: FaceLandmark) -> float:
    return float(landmark.session.run(None, {landmark.input_name: tensor[np.newaxis]})[1].flatten()[-1])


def analyse(det_img: Image.Image, lm_img: Image.Image, detector: FaceDetection, landmark: FaceLandmark) -> dict:
    """Run the detect_landmarks steps; report on the first (main) detection."""
    size = det_img.size
    detections = detector(det_img)
    faces = [m for m in (landmark(lm_img, face_detection_to_roi(d, size)) for d in detections) if m]
    outcome = 'no_face' if not faces else 'multiple_faces' if len(faces) > 1 else 'ok'
    roi = face_detection_to_roi(detections[0], size)
    tensor = landmark_input(lm_img, roi, landmark)
    logit = presence_logit(tensor, landmark)
    inner = tensor[48:144, 48:144] * 255  # central face region of the crop
    rgb = inner.reshape(-1, 3).mean(0)
    w, h = size
    return {
        'detections': detections,
        'roi': roi,
        'tensor': tensor,
        'det_score': float(detections[0].score),
        'logit': logit,
        'presence': float(sigmoid(np.array([logit]))[0]),
        'landmarks': landmark(lm_img, roi),
        'points': np.array([(m.x * w, m.y * h) for m in faces[0]]) if outcome == 'ok' else None,
        'outcome': outcome,
        'face_rgb': rgb,
        'g_over_r': rgb[1] / rgb[0],
        'lum_std': float((inner @ [0.299, 0.587, 0.114]).std()),
    }


def save_images(name: str, det_img: Image.Image, frame: Image.Image, r: dict) -> None:
    # 1. what BlazeFace sees: bbox + keypoints + rotated ROI
    img = render_to_image(
        detections_to_render_data(r['detections'], bounds_color=Colors.GREEN, keypoint_color=Colors.RED, line_width=3, point_width=6),
        det_img.copy(),
    )
    ImageDraw.Draw(img).polygon([tuple(p) for p in r['roi'].scaled(det_img.size).points()], outline=(0, 128, 255), width=3)
    img.save(IMG_DIR / f'{name}_1_detection.jpg', quality=88)

    # 2. the exact 192x192 crop the landmark model receives, shown 2x
    crop = Image.fromarray((r['tensor'] * 255).astype(np.uint8))
    crop.resize((384, 384), Image.NEAREST).save(IMG_DIR / f'{name}_2_landmark_input.png')

    # 3. landmark model verdict, drawn on the original frame
    ok = bool(r['landmarks'])
    colour = GREEN if ok else RED
    canvas = frame.copy()
    if ok:
        canvas = render_to_image(face_landmarks_to_render_data(r['landmarks'], Colors.RED, Colors.GREEN, 2), canvas)
    draw = ImageDraw.Draw(canvas, 'RGBA')
    font, small = ImageFont.truetype('arialbd.ttf', 26), ImageFont.truetype('arial.ttf', 20)
    if not ok:
        corners = [tuple(p) for p in r['roi'].scaled(frame.size).points()]
        draw.polygon(corners, outline=RED, width=5, fill=(230, 40, 40, 50))
        draw.line([corners[0], corners[2]], fill=RED, width=4)
        draw.line([corners[1], corners[3]], fill=RED, width=4)
        cx, cy = sum(p[0] for p in corners) / 4, sum(p[1] for p in corners) / 4
        for i, (label, f) in enumerate([('REJECTED', font), ('face presence too low', small)]):
            tw, y = draw.textlength(label, font=f), cy - 34 + i * 40
            draw.rectangle([cx - tw / 2 - 10, y - 4, cx + tw / 2 + 10, y + 32], fill=(0, 0, 0, 190))
            draw.text((cx - tw / 2, y), label, font=f, fill=RED)
    x0, y0, bw = 20, 20, 460
    draw.rectangle([x0 - 10, y0 - 10, x0 + bw + 10, y0 + 128], fill=(0, 0, 0, 200))
    draw.text((x0, y0), 'Landmark model: face presence', font=small, fill='white')
    p = r['presence']
    draw.text((x0, y0 + 26), f'{p:.6f}' if p < 0.001 else f'{p:.4f}', font=font, fill=colour)
    draw.text((x0 + 200, y0 + 32), f'(logit {r["logit"]:+.2f})', font=small, fill=(200, 200, 200))
    by = y0 + 66
    draw.rectangle([x0, by, x0 + bw, by + 18], outline='white', width=2)
    draw.rectangle([x0, by, x0 + max(3, int(bw * p)), by + 18], fill=colour)
    tx = x0 + int(bw * DETECTION_THRESHOLD)
    draw.line([tx, by - 6, tx, by + 24], fill='white', width=3)
    verdict = f'≥ threshold {DETECTION_THRESHOLD} → {len(r["landmarks"])} landmarks' if ok else f'< threshold {DETECTION_THRESHOLD} → rejected, 0 landmarks'
    draw.text((x0, by + 28), verdict, font=small, fill=colour)
    canvas.save(IMG_DIR / f'{name}_3_result.jpg', quality=88)


if __name__ == '__main__':
    frame = Image.open(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FRAME).convert('RGB')
    detector = FaceDetection(model_type=FaceDetectionModel.SHORT)
    landmark = FaceLandmark()
    IMG_DIR.mkdir(exist_ok=True)

    # (BlazeFace input, landmark model input); images are only saved for the ones shown in the report
    shown = ('original', 'grayscale', 'colour_det_gray_crop')
    variants = {
        'original': (frame, frame),
        'grayscale': (grayscale(frame), grayscale(frame)),
        'whitebalance': (white_balance(frame), white_balance(frame)),
        'autocontrast': (ImageOps.autocontrast(frame, cutoff=1),) * 2,
        'colour_det_gray_crop': (frame, grayscale(frame)),
    }
    # extra variants to isolate the cause (no images)
    isolation = {
        'contrast_x2': ImageEnhance.Contrast(frame).enhance(2.0),
        'brightness_x1.5': ImageEnhance.Brightness(frame).enhance(1.5),
        'desaturate_50pct': ImageEnhance.Color(frame).enhance(0.5),
        'equalize': ImageOps.equalize(frame),
    }

    results = {}
    print(f'{"variant":22s} {"faces":>5s} {"det":>6s} {"logit":>7s} {"presence":>9s} {"outcome":>14s}  face RGB mean     G/R   lum std')
    for name, (det_img, lm_img) in {**variants, **{k: (v, v) for k, v in isolation.items()}}.items():
        r = results[name] = analyse(det_img, lm_img, detector, landmark)
        rgb = ', '.join(f'{c:.0f}' for c in r['face_rgb'])
        print(f'{name:22s} {len(r["detections"]):5d} {r["det_score"]:6.3f} {r["logit"]:+7.2f} {r["presence"]:9.6f} {r["outcome"]:>14s}  '
              f'({rgb:13s})  {r["g_over_r"]:.2f}  {r["lum_std"]:.1f}')
        if name in shown:
            save_images(name, det_img, frame, r)

    # does grayscaling the detector input as well change the landmarks?
    a, b = results['colour_det_gray_crop'], results['grayscale']
    roi_a, roi_b = a['roi'].scaled(frame.size), b['roi'].scaled(frame.size)
    iod = np.linalg.norm(b['points'][LEFT_EYE_OUTER] - b['points'][RIGHT_EYE_OUTER])
    diff = np.linalg.norm(a['points'] - b['points'], axis=1)
    print(f'\ncolour_det_gray_crop vs grayscale: ROI centre shift {np.hypot(roi_a.x_center - roi_b.x_center, roi_a.y_center - roi_b.y_center):.1f}px, '
          f'size {roi_a.width:.0f} vs {roi_b.width:.0f}px | landmark diff mean {diff.mean():.2f}px, max {diff.max():.2f}px '
          f'({diff.mean() / iod * 100:.2f}% of IOD = {iod:.0f}px)')
