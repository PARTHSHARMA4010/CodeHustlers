import os
import cv2
import numpy as np
import requests
import base64
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
from mediapipe import solutions
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
from data import dataset  # Assuming data.py contains the dataset
from io import BytesIO
from collections import Counter

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- Configuration and Model Setup ---

# Google Generative AI Configuration
# genai.configure(api_key='AIzaSyDAZNIlG2P3mSmrHY6QKqrpOntDISypHIQ')  # Use environment variable for API key
# generation_config = {
#     "temperature": 0.1,
#     "top_p": 0.95,
#     "top_k": 40,
#     "max_output_tokens": 8192,
#     "response_mime_type": "application/json",
# }
# gemini_model = genai.GenerativeModel(
#     model_name="gemini-2.0-flash-exp",
#     generation_config=generation_config,
# )
# gemini_chat_history = []

# Random Forest Model and Data Processing
def process_data(data, for_prediction=False):
    tone_map = {"light": 0, "medium": 1, "dark": 2}
    texture_map = {"smooth": 0, "normal": 1, "dry": 2, "oily": 3, "rough": 4}
    dark_circles_map = {"absent": 0, "present": 1, "no": 0}

    features = []
    labels = []
    outputs_map = {}

    for entry in data:
        skin_analysis = entry["skin_analysis"]
        nail_analysis = entry["nail_analysis"]

        tone = tone_map.get(skin_analysis["tone"], 0)
        texture = texture_map.get(skin_analysis["texture"], 1)
        dark_circles = dark_circles_map.get(skin_analysis["dark_circles"], 0)
        spot_count = len(skin_analysis.get("spots", []))
        avg_spot_size = np.mean([spot["size"] for spot in skin_analysis.get("spots", [])]) if spot_count > 0 else 0

        nail_color = nail_analysis.get("color", {}).get("RGB", [0, 0, 0])
        edge_intensity = nail_analysis.get("texture", {}).get("edge_intensity", 0)
        shapes_detected = nail_analysis.get("shape", {}).get("shapes_detected", [])
        round_shape = 1 if "Round" in shapes_detected else 0
        irregular_shape = 1 if "Irregular" in shapes_detected else 0

        feature = [
            tone, texture, dark_circles, spot_count, avg_spot_size,
            nail_color[0], nail_color[1], nail_color[2], edge_intensity,
            round_shape, irregular_shape
        ]
        features.append(feature)

        if not for_prediction:
            outputs = entry["outputs"]
            labels.append(outputs["disease"])
            outputs_map[outputs["disease"]] = outputs

    feature_columns = [
        'tone', 'texture', 'dark_circles', 'spot_count', 'avg_spot_size',
        'color_R', 'color_G', 'color_B', 'edge_intensity',
        'round_shape', 'irregular_shape'
    ]

    if for_prediction:
        return pd.DataFrame(features, columns=feature_columns), None, None

    return pd.DataFrame(features, columns=feature_columns), labels, outputs_map

# Initialize the Random Forest model, label encoder, and outputs map
X, y, outputs_map = process_data(dataset)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y_encoded)

# --- Image and Video Analysis Functions ---

def read_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode the image from the URL: {url}")
        return image
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching image from URL: {url}, Error: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")

def base64_to_array(base64_string):
    try:
        # Remove the data URL prefix if it exists
        if ';base64,' in base64_string:
            base64_string = base64_string.split(';base64,')[-1]
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode base64 string to image")
        return img
    except Exception as e:
        raise ValueError(f"Error decoding base64 string: {e}")

def analyze_skin(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)
        if not results.detections:
            raise ValueError("No face detected in the image")

        face = results.detections[0]
        bbox = face.location_data.relative_bounding_box
        h, w, _ = image.shape
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        face_image = image_rgb[y1:y2, x1:x2]

    features = {
        "tone": detect_skin_tone(face_image),
        "texture": detect_skin_texture(face_image),
        "dark_circles": detect_dark_circles(image_rgb, bbox),
        "spots": detect_skin_spots(face_image)
    }
    return features

def detect_skin_tone(face_image):
    hsv_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
    avg_hue = np.mean(hsv_image[:, :, 0])
    if avg_hue < 15:
        return "light"
    elif avg_hue < 30:
        return "medium"
    else:
        return "dark"

def detect_skin_texture(face_image):
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    lbp = cv2.Laplacian(gray_image, cv2.CV_64F)
    texture_score = np.var(lbp)
    if texture_score < 100:
        return "smooth"
    elif texture_score < 500:
        return "normal"
    else:
        return "rough"

def detect_dark_circles(image, bbox):
    h, w, _ = image.shape
    y1 = int(bbox.ymin * h)
    x1 = int(bbox.xmin * w)
    box_height = int(bbox.height * h)
    eye_region_y_start = y1 + int(0.3 * box_height)
    eye_region_y_end = y1 + int(0.5 * box_height)
    eye_region = image[eye_region_y_start:eye_region_y_end, x1:]
    if eye_region.size == 0:
        return "absent"
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
    avg_brightness = np.mean(gray_eye)
    return "present" if avg_brightness < 50 else "absent"

def detect_skin_spots(face_image):
    blurred = cv2.GaussianBlur(face_image, (15, 15), 0)
    diff = cv2.absdiff(face_image, blurred)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spots = []
    for c in contours:
        if c.shape[0] > 0:
            x, y = c[0][0]
            spots.append({"location": f"x={x}, y={y}", "size": len(c)})

    return spots if spots else []

def analyze_nails(image_array):
    image = cv2.resize(image_array, (600, 400))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    nail_segment = cv2.bitwise_and(image, image, mask=mask)

    dominant_color = extract_dominant_color(nail_segment)
    texture_features = analyze_texture(nail_segment)
    shape_features = detect_nail_shape(nail_segment)

    return {
        "color": dominant_color,
        "texture": texture_features,
        "shape": shape_features,
    }

def extract_dominant_color(image, clusters=3):
    pixels = image.reshape(-1, 3)
    pixels = pixels[pixels.sum(axis=1) > 0]
    kmeans = KMeans(n_clusters=clusters, random_state=42).fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    counts = np.bincount(kmeans.labels_)
    dominant_color = dominant_colors[np.argmax(counts)]

    return {
        "RGB": tuple(map(int, dominant_color)),
        "Hex": "#{:02x}{:02x}{:02x}".format(*map(int, dominant_color))
    }

def analyze_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
    texture_score = np.sum(hist[50:150]) / np.sum(hist)
    return {"edge_intensity": round(float(texture_score), 2)}

def detect_nail_shape(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nail_shapes = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            if len(approx) >= 8:
                nail_shapes.append("Oval")
            elif 5 <= len(approx) < 8:
                nail_shapes.append("Round")
            else:
                nail_shapes.append("Irregular")

    return {"shapes_detected": nail_shapes}

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")

    frame_count = 0
    skin_results = []
    nail_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        try:
            skin_features = analyze_skin(frame)  # Pass the frame array directly
            skin_results.append(skin_features)
        except ValueError as e:
            print(f"Skipping skin analysis for frame {frame_count} due to error: {e}")

        try:
            nail_features = analyze_nails(frame)  # Pass the frame array directly
            nail_results.append(nail_features)
        except ValueError as e:
            print(f"Skipping nail analysis for frame {frame_count} due to error: {e}")

    cap.release()

    # Aggregate skin analysis results
    avg_skin_tone = Counter([res["tone"] for res in skin_results if "tone" in res]).most_common(1)[0][0] if skin_results and any("tone" in res for res in skin_results) else None
    avg_texture = Counter([res["texture"] for res in skin_results if "texture" in res]).most_common(1)[0][0] if skin_results and any("texture" in res for res in skin_results) else None
    all_spots = [spot for res in skin_results for spot in res.get("spots", [])]
    dark_circles_present = any(res.get("dark_circles") == "present" for res in skin_results)

    aggregated_skin_analysis = {
        "tone": avg_skin_tone,
        "texture": avg_texture,
        "dark_circles": "present" if dark_circles_present else "absent",
        "spots": all_spots
    }

    # Aggregate nail analysis results
    all_nail_colors = [tuple(res["color"]["RGB"]) for res in nail_results if res and "color" in res]
    most_frequent_nail_color = {}
    if all_nail_colors:
        color_counts = Counter(all_nail_colors)
        most_frequent_color_rgb = color_counts.most_common(1)[0][0]
        most_frequent_nail_color = {
            "RGB": list(most_frequent_color_rgb),
            "Hex": "#{:02x}{:02x}{:02x}".format(*most_frequent_color_rgb)
        }

    all_nail_shapes = [shape for res in nail_results if res and "shape" in res for shape in res["shape"].get("shapes_detected", [])]
    unique_nail_shapes = list(set(all_nail_shapes))

    aggregated_nail_analysis = {
        "color": most_frequent_nail_color,
        "texture": {"edge_intensity": np.mean([res["texture"]["edge_intensity"] for res in nail_results if res and "texture" in res]) if any(res and "texture" in res for res in nail_results) else 0},
        "shape": {"shapes_detected": unique_nail_shapes}
    }

    return {
        "skin_analysis": aggregated_skin_analysis,
        "nail_analysis": aggregated_nail_analysis
    }

# --- API Endpoints ---

def make_prediction(analysis_result):
    input_features, _, _ = process_data([analysis_result], for_prediction=True)
    predicted_label = rf_model.predict(input_features)
    predicted_disease = label_encoder.inverse_transform(predicted_label)[0]
    return {
        "predicted_disease": predicted_disease,
        "report": outputs_map.get(predicted_disease, {})
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        val_type = data.get('Type')
        url = data.get('Url')

        if not val_type or not url:
            return jsonify({"error": "Both 'Type' and 'Url' fields are required"}), 400

        if any(val_type.lower().endswith(i) for i in ['png', 'jpeg', 'jpg']):
            analysis_result = {}
            if 'base64' in url:
                try:
                    image_array = base64_to_array(url)
                    analysis_result["skin_analysis"] = analyze_skin(image_array)
                    analysis_result["nail_analysis"] = analyze_nails(image_array)
                except ValueError as e:
                    return jsonify({"error": f"Error processing base64 image: {str(e)}"}), 400
            else:
                try:
                    image = read_image_from_url(url)
                    analysis_result["skin_analysis"] = analyze_skin(image)
                    analysis_result["nail_analysis"] = analyze_nails(image)
                except ValueError as e:
                    return jsonify({"error": f"Error processing image URL: {str(e)}"}), 400
            prediction_result = make_prediction(analysis_result)
            return jsonify(prediction_result), 200

        elif val_type.lower().endswith('mp4'):
            try:
                video_path = download_file(url)
                analysis_result = analyze_video(video_path)
                os.remove(video_path)
                prediction_result = make_prediction(analysis_result)
                return jsonify(prediction_result), 200
            except ValueError as e:
                return jsonify({"error": f"Error processing video: {str(e)}"}), 400

        else:
            return jsonify({"error": "Unsupported 'Type' provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    chat_session = gemini_model.start_chat(history=gemini_chat_history)
    response = chat_session.send_message(user_input)
    model_response = response.text

    gemini_chat_history.append({"role": "user", "parts": [user_input]})
    gemini_chat_history.append({"role": "model", "parts": [model_response]})

    return jsonify({"response": model_response})

# --- Helper Functions ---

def download_file(url):
    """Downloads a file from a URL and returns the local file path."""
    try:
        local_filename = 'test.mp4'
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error downloading file from URL: {url}, Error: {e}")

# --- Main Execution ---

if __name__ == '__main__':
    app.run(debug=True, port=5001)