import cv2
import torch
import numpy as np
from ultralytics import YOLO
import sqlite3
import os
from datetime import datetime
import time
import uuid
import logging
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Configuration parameters
CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence score for a detection to be considered valid
PROCESS_INTERVAL = 1.5  # Time interval (in seconds) between processing frames for detection
CAMERA_INDEX = 1  # Index of the camera to use (0 for default, 1 for external camera)
MODEL_PATH = 'best.pt'  # Path to the trained YOLO model file
LANE_WIDTH_METERS = 3.7  # Assumed width of a road lane in meters
SPEED_THRESHOLDS = [10, 20, 30]  # Speed thresholds in km/h for categorizing vehicle speed
OUTPUT_SPEED_FACTOR = 0.75  # Factor to slow down the output video (0.75 = 75% of original speed)
VERTICAL_THRESHOLD = 0.3  # Vertical threshold for considering potholes (0.5 = middle of the image)

# Set up logging
logging.basicConfig(filename='pothole_detection.log', level=logging.INFO)

# Load the trained model
model = YOLO(MODEL_PATH)

# Create necessary directories
for directory in ['VIDEOS', 'DATA', 'pothole_frames', 'Analysis']:
    if not os.path.exists(directory):
        os.makedirs(directory)
pytho
# Create a new database connection
conn = sqlite3.connect('potholes.db')
c = conn.cursor()

# Drop the existing table if it exists and create a new one
c.execute('DROP TABLE IF EXISTS potholes')
c.execute('''CREATE TABLE potholes
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              frame_number INTEGER,
              timestamp TEXT,
              image_path TEXT,
              very_big_count INTEGER,
              big_count INTEGER,
              medium_count INTEGER,
              small_count INTEGER,
              very_small_count INTEGER,
              central_count INTEGER,
              sided_count INTEGER,
              out_of_road_count INTEGER,
              gravity_score REAL)''')

def bird_eye_transform(img):
    height, width = img.shape[:2]
    src_points = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
    dst_points = np.float32([[width*0.05, height], [width*0.95, height], [width*0.75, 0], [width*0.25, 0]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)

def get_size_category(area, frame_area):
    ratio = area / frame_area
    if ratio > 0.05:
        return "Very Big", (255, 0, 0)  # Blue
    elif ratio > 0.02:
        return "Big", (0, 255, 0)  # Green
    elif ratio > 0.005:
        return "Medium", (0, 255, 255)  # Yellow
    elif ratio > 0.0005:
        return "Small", (0, 165, 255)  # Orange
    else:
        return "Very Small", (0, 0, 255)  # Red

def get_road_position(x, frame_width):
    ratio = x / frame_width
    if ratio < 0.1 or ratio > 0.9:
        return "Out of Road"
    elif 0.45 <= ratio <= 0.55:
        return "Central"
    else:
        return "Sided"

def draw_boxes(img, boxes, labels, confidences, frame_number, timestamp):
    frame_area = img.shape[0] * img.shape[1]
    frame_width = img.shape[1]
    frame_height = img.shape[0]
    frame_pothole_counts = {"Very Big": 0, "Big": 0, "Medium": 0, "Small": 0, "Very Small": 0}
    frame_position_counts = {"Central": 0, "Sided": 0, "Out of Road": 0}
    
    valid_boxes = []
    valid_labels = []
    valid_confidences = []
    
    for box, label, conf in zip(boxes, labels, confidences):
        if conf > CONFIDENCE_THRESHOLD:
            x, y, w, h = box
            center_y = y / frame_height
            
            if center_y > VERTICAL_THRESHOLD:
                valid_boxes.append(box)
                valid_labels.append(label)
                valid_confidences.append(conf)
                
                area = w * h
                size_category, color = get_size_category(area, frame_area)
                position = get_road_position(x, frame_width)
                
                x_min, y_min = int(x - w / 2), int(y - h / 2)
                x_max, y_max = int(x + w / 2), int(y + h / 2)
                
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
                cv2.putText(img, f"{size_category} ({conf:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                frame_pothole_counts[size_category] += 1
                frame_position_counts[position] += 1
    
    return img, frame_pothole_counts, frame_position_counts, valid_boxes, valid_labels, valid_confidences

def calculate_gravity_score(pothole_counts, position_counts):
    size_weights = {"Very Big": 5, "Big": 4, "Medium": 3, "Small": 2, "Very Small": 1}
    position_weights = {"Central": 3, "Sided": 2, "Out of Road": 1}
    
    size_score = sum(count * size_weights[size] for size, count in pothole_counts.items())
    position_score = sum(count * position_weights[pos] for pos, count in position_counts.items())
    
    total_potholes = sum(pothole_counts.values())
    return (size_score + position_score) / total_potholes if total_potholes > 0 else 0

def apply_fixed_info(img, cumulative_pothole_counts, cumulative_position_counts, timestamp, fps, estimated_speed):
    height, width = img.shape[:2]
    
    # Timestamp (bottom left)
    cv2.putText(img, timestamp, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Total count (top center, large and red)
    total_count = sum(cumulative_pothole_counts.values())
    total_count_text = f"Total: {total_count}"
    text_size = cv2.getTextSize(total_count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (width - text_size[0]) // 2
    cv2.putText(img, total_count_text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Sizes (left side, white)
    y_offset = 30
    for size, count in cumulative_pothole_counts.items():
        cv2.putText(img, f"{size}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
    
    # Positions (right side, white)
    y_offset = 30
    for position, count in cumulative_position_counts.items():
        text = f"{position}: {count}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        x_position = width - text_size[0] - 10
        cv2.putText(img, text, (x_position, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
    
    return img

def estimate_speed(pixels_moved, fps, pixels_per_meter):
    if fps == 0:
        return 0
    meters_per_second = (pixels_moved / pixels_per_meter) * fps
    km_per_hour = meters_per_second * 3.6
    return min(SPEED_THRESHOLDS, key=lambda x: abs(x - km_per_hour))

def perform_analysis():
    conn = sqlite3.connect('potholes.db')
    df = pd.read_sql_query("SELECT * FROM potholes", conn)
    conn.close()

    if df.empty:
        print("No potholes detected in the video.")
        return

    # Generate analysis report
    report = "Pothole Detection Analysis Report\n"
    report += "================================\n\n"
    report += f"Total frames with potholes: {len(df)}\n"
    report += f"Total potholes detected: {df[['very_big_count', 'big_count', 'medium_count', 'small_count', 'very_small_count']].sum().sum()}\n\n"

    report += "Pothole size distribution:\n"
    for size in ['very_big_count', 'big_count', 'medium_count', 'small_count', 'very_small_count']:
        report += f"  {size}: {df[size].sum()}\n"

    report += "\nPothole position distribution:\n"
    for position in ['central_count', 'sided_count', 'out_of_road_count']:
        report += f"  {position}: {df[position].sum()}\n"

    report += f"\nAverage gravity score: {df['gravity_score'].mean():.2f}\n"
    report += f"Max gravity score: {df['gravity_score'].max():.2f}\n"
    report += f"Min gravity score: {df['gravity_score'].min():.2f}\n"

    report += "\nFrames with highest gravity scores:\n"
    high_gravity_frames = df.nlargest(5, 'gravity_score')[['frame_number', 'timestamp', 'gravity_score', 'image_path']]
    report += high_gravity_frames.to_string(index=False)

    # Save the report
    with open('Analysis/pothole_analysis_report.txt', 'w') as f:
        f.write(report)

    # Generate plots
    plt.figure(figsize=(12, 6))
    plt.plot(df['frame_number'], df[['very_big_count', 'big_count', 'medium_count', 'small_count', 'very_small_count']].sum(axis=1))
    plt.title('Pothole Distribution Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Potholes')
    plt.savefig('Analysis/potholes_over_time.png')
    plt.close()

    size_counts = df[['very_big_count', 'big_count', 'medium_count', 'small_count', 'very_small_count']].sum()
    plt.figure(figsize=(10, 6))
    size_counts.plot(kind='bar')
    plt.title('Distribution of Pothole Sizes')
    plt.xlabel('Size Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('Analysis/pothole_size_distribution.png')
    plt.close()

    position_counts = df[['central_count', 'sided_count', 'out_of_road_count']].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(position_counts, labels=position_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Pothole Positions')
    plt.axis('equal')
    plt.savefig('Analysis/pothole_position_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(df['gravity_score'], bins=20)
    plt.title('Distribution of Gravity Scores')
    plt.xlabel('Gravity Score')
    plt.ylabel('Frequency')
    plt.savefig('Analysis/gravity_score_distribution.png')
    plt.close()

    print("Analysis complete. Files saved in the 'Analysis' folder:")
    print("- pothole_analysis_report.txt")
    print("- potholes_over_time.png")
    print("- pothole_size_distribution.png")
    print("- pothole_position_distribution.png")
    print("- gravity_score_distribution.png")

if __name__ == "__main__":
    # Set up video capture from external camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open external camera (index {CAMERA_INDEX}).")
        print("Trying to open the default camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any camera.")
            exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Set a default value if fps is 0

    print(f"Camera opened successfully. Resolution: {frame_width}x{frame_height}, FPS: {fps}")

    # Create VideoWriter objects with adjusted FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = fps * OUTPUT_SPEED_FACTOR
    out_original = cv2.VideoWriter(os.path.join('VIDEOS', 'output_video_original.mp4'), fourcc, output_fps, (frame_width, frame_height))
    out_boxes = cv2.VideoWriter(os.path.join('VIDEOS', 'output_video_boxes.mp4'), fourcc, output_fps, (frame_width, frame_height))
    out_bird_eye = cv2.VideoWriter(os.path.join('VIDEOS', 'output_video_bird_eye.mp4'), fourcc, output_fps, (frame_width, frame_height))

    # Initialize variables
    frame_count = 0
    prev_frame_time = 0
    last_process_time = time.time()
    cumulative_pothole_counts = {"Very Big": 0, "Big": 0, "Medium": 0, "Small": 0, "Very Small": 0}
    cumulative_position_counts = {"Central": 0, "Sided": 0, "Out of Road": 0}
    prev_frame = None
    estimated_speed = 0

    # Create CSV files for pothole data
    csv_file_path = os.path.join('DATA', 'pothole_data.csv')
    csv_headers = ['frame_number', 'timestamp', 'image_path', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'confidence', 'size_category', 'position', 'gravity_score']

    frame_box_csv_path = os.path.join('DATA', 'frame_box_data.csv')
    frame_box_headers = ['frame_filename', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'confidence']

    with open(csv_file_path, 'w', newline='') as csvfile, open(frame_box_csv_path, 'w', newline='') as frame_box_csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        csv_writer.writeheader()
        frame_box_writer = csv.DictWriter(frame_box_csvfile, fieldnames=frame_box_headers)
        frame_box_writer.writeheader()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Update timestamp for every frame
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Calculate FPS
                if prev_frame_time != 0:
                    fps = 1/(current_time - prev_frame_time)
                prev_frame_time = current_time

                # Estimate speed
                pixels_per_meter = frame.shape[1] / LANE_WIDTH_METERS
                if prev_frame is not None:
                    frame_diff = cv2.absdiff(frame, prev_frame)
                    pixels_moved = np.sum(frame_diff) / 255
                    estimated_speed = estimate_speed(pixels_moved, fps, pixels_per_meter)
                prev_frame = frame.copy()

                # Create copies for different outputs
                frame_with_boxes = frame.copy()
                bird_eye_frame = bird_eye_transform(frame)

                # Process frames at regular intervals
                if current_time - last_process_time >= PROCESS_INTERVAL:
                    last_process_time = current_time

                    # Perform inference
                    results = model(frame, conf=CONFIDENCE_THRESHOLD)
                    
                    frame_pothole_counts = {"Very Big": 0, "Big": 0, "Medium": 0, "Small": 0, "Very Small": 0}
                    frame_position_counts = {"Central": 0, "Sided": 0, "Out of Road": 0}
                    
                    for r in results:
                        boxes = r.boxes
                        if len(boxes) > 0:
                            bboxes = boxes.xywh.cpu().numpy()
                            scores = boxes.conf.cpu().numpy()
                            class_ids = boxes.cls.cpu().numpy().astype(int)
                            labels = [model.names[c] for c in class_ids]

                            frame_with_boxes, frame_pothole_counts, frame_position_counts, valid_boxes, valid_labels, valid_confidences = draw_boxes(
                                frame_with_boxes, bboxes, labels, scores, frame_count, timestamp)
                            
                            bird_eye_frame, _, _, _, _, _ = draw_boxes(
                                bird_eye_frame, bboxes, labels, scores, frame_count, timestamp)

                            # Update cumulative counts
                            for size, count in frame_pothole_counts.items():
                                cumulative_pothole_counts[size] += count
                            for position, count in frame_position_counts.items():
                                cumulative_position_counts[position] += count

                            # Save frame and data if potholes are detected
                            if sum(frame_pothole_counts.values()) > 0:
                                frame_id = str(uuid.uuid4())
                                frame_name = f"{timestamp.replace(':', '-')}_{frame_id}"
                                
                                frame_path_with_boxes = os.path.join('pothole_frames', f"{frame_name}_with_boxes.jpg")
                                cv2.imwrite(frame_path_with_boxes, frame_with_boxes)
                                
                                frame_path_without_boxes = os.path.join('DATA', f"{frame_name}.jpg")
                                cv2.imwrite(frame_path_without_boxes, frame)
                                
                                # Insert into database
                                c.execute("""INSERT INTO potholes 
                                            (frame_number, timestamp, image_path, 
                                            very_big_count, big_count, medium_count, small_count, very_small_count,
                                            central_count, sided_count, out_of_road_count, gravity_score) 
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        (frame_count, timestamp, frame_path_with_boxes, 
                                        frame_pothole_counts["Very Big"], frame_pothole_counts["Big"], 
                                        frame_pothole_counts["Medium"], frame_pothole_counts["Small"], 
                                        frame_pothole_counts["Very Small"],
                                        frame_position_counts["Central"], frame_position_counts["Sided"], 
                                        frame_position_counts["Out of Road"], calculate_gravity_score(frame_pothole_counts, frame_position_counts)))
                                
                                # Write to CSV
                                for box, score, label in zip(valid_boxes, valid_confidences, valid_labels):
                                    x, y, w, h = box
                                    size_category, _ = get_size_category(w * h, frame.shape[0] * frame.shape[1])
                                    position = get_road_position(x, frame.shape[1])
                                    csv_writer.writerow({
                                        'frame_number': frame_count,
                                        'timestamp': timestamp,
                                        'image_path': frame_path_without_boxes,
                                        'bbox_x': x,
                                        'bbox_y': y,
                                        'bbox_width': w,
                                        'bbox_height': h,
                                        'confidence': score,
                                        'size_category': size_category,
                                        'position': position,
                                        'gravity_score': calculate_gravity_score(frame_pothole_counts, frame_position_counts)
                                    })
                                    
                                    # Write to frame_box CSV
                                    frame_box_writer.writerow({
                                        'frame_filename': os.path.basename(frame_path_without_boxes),
                                        'bbox_x': x,
                                        'bbox_y': y,
                                        'bbox_width': w,
                                        'bbox_height': h,
                                        'confidence': score
                                    })

                # Apply fixed information to all frames
                frame_with_info = apply_fixed_info(frame.copy(), cumulative_pothole_counts, cumulative_position_counts, timestamp, fps, estimated_speed)
                frame_with_boxes_info = apply_fixed_info(frame_with_boxes, cumulative_pothole_counts, cumulative_position_counts, timestamp, fps, estimated_speed)
                bird_eye_frame_with_info = apply_fixed_info(bird_eye_frame, cumulative_pothole_counts, cumulative_position_counts, timestamp, fps, estimated_speed)

                # Write frames to video outputs
                out_original.write(frame_with_info)
                out_boxes.write(frame_with_boxes_info)
                out_bird_eye.write(bird_eye_frame_with_info)

                # Display the frame
                cv2.imshow('Pothole Detection', frame_with_boxes_info)

                # Press Q on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nScript interrupted by user.")

        finally:
            # Release everything
            cap.release()
            out_original.release()
            out_boxes.release()
            out_bird_eye.release()
            cv2.destroyAllWindows()

            # Commit changes and close the database connection
            conn.commit()
            conn.close()

            print(f"\nProcessed {frame_count} frames")
            print("Output videos saved in the 'VIDEOS' folder:")
            print("- output_video_original.mp4 (Original video with info overlay)")
            print("- output_video_boxes.mp4 (Video with bounding boxes and info overlay)")
            print("- output_video_bird_eye.mp4 (Bird's eye view with bounding boxes and info overlay)")
            print("Pothole database saved as 'potholes.db'")
            print("Pothole frames with boxes saved in 'pothole_frames' directory")
            print("Pothole frames without boxes and CSV data saved in 'DATA' directory")

            # Perform the analysis
            perform_analysis()