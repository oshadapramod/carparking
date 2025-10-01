# Parking Occupancy Detection System

A computer vision-based parking lot monitoring system that automatically detects and tracks parking space availability using drone footage.

## Overview

This project implements an automated parking occupancy detection system using OpenCV and traditional computer vision techniques. The system processes aerial video footage to identify occupied and vacant parking spaces in real-time, providing valuable insights for parking management.

## Features

- **Automated Slot Detection**: Load pre-defined parking slot coordinates from JSON
- **Real-time Occupancy Detection**: Process video footage to identify occupied/vacant spaces
- **Visual Feedback**: Color-coded bounding boxes (Green=Available, Red=Occupied)
- **Statistical Analysis**: Track occupancy rates over time
- **Calibration Tool**: Analyze and optimize detection thresholds
- **Video Output**: Generate annotated video with detection overlays

## Technical Approach

### Detection Method
The system uses a **simple yet effective** computer vision approach combining:
- **Edge Detection**: Canny edge detection to identify vehicle boundaries
- **Texture Analysis**: Variance and standard deviation to distinguish textured surfaces (cars) from empty pavement
- **Frame Differencing**: Track changes between frames to optimize processing

### Why This Approach?
- No dependency on large deep learning models or GPUs
- Works reliably with aerial/drone footage
- Handles various car colors including dark vehicles
- Fast processing suitable for real-time applications
- Transparent and explainable detection logic

## Dataset

- **Source**: Drone footage (DJI_0012.MOV)
- **Resolution**: 3840x2160 (4K)
- **Duration**: 452.4 seconds
- **Frame Rate**: 25 FPS
- **Total Frames**: 11,309
- **Parking Slots**: 312 pre-annotated slots

The dataset consists of aerial footage of a commercial parking lot with diverse vehicle types, colors, and lighting conditions.

## Installation

### Prerequisites
```bash
Python 3.7+
OpenCV (cv2)
NumPy
```

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd carparking

# Install dependencies
pip install opencv-python numpy

# Verify files are present
# - carparking.py
# - parking_slots.json
# - DJI_0012.MOV
```

## Usage

### Basic Usage
```bash
python carparking.py
```

### Interactive Workflow
The script provides an interactive workflow:

1. **Load Parking Slots**: Automatically loads 312 parking slot coordinates
2. **Visualize Slots** (Optional): Preview slot bounding boxes on reference frame
3. **Calibrate Thresholds** (Recommended): Analyze slot features to optimize detection
4. **Process Video**: Run detection on full video with customizable parameters

### Example Session
```
Do you want to visualize loaded slots? (y/n): y
Do you want to analyze and calibrate thresholds? (recommended) (y/n): y
Use suggested thresholds? (y/n): y
Process video now? (y/n): y
Check slots every N frames (default 30): 30
Show real-time preview? (y/n): y
```

## Results

### Detection Performance
- **Average Occupancy**: 65.2%
- **Max Occupancy**: 70.2%
- **Min Occupancy**: 57.4%
- **Processing Speed**: ~377 frames per second (with step=30)

### Output Files
```
parking_output/
├── output.mp4                    # Annotated video with bounding boxes
├── occupancy_stats.json          # Frame-by-frame statistics
├── loaded_slots_preview.jpg      # Slot visualization
└── frames/
    └── frame_000000.jpg          # Reference frame
```

## Model Implementation

### Algorithm Pipeline
1. **Slot Loading**: Load predefined slot coordinates from JSON
2. **Frame Preprocessing**: Convert to grayscale, apply Gaussian blur
3. **Feature Extraction**:
   - Canny edge detection (thresholds: 40, 120)
   - Calculate edge density
   - Compute variance and standard deviation
4. **Classification**:
   - Occupied if: edge_density > threshold OR (variance > threshold AND std_dev > 25)
   - Empty otherwise
5. **Optimization**: Only re-check slots that show significant change

### Configurable Parameters
- `edge_threshold`: Edge density threshold (default: 0.08, calibrated: ~0.11)
- `variance_threshold`: Texture variance threshold (default: 2500, calibrated: ~1508)
- `step`: Frame check interval (default: 30 frames)

## Evaluation Metrics

The system tracks:
- **Available Slots**: Number of empty parking spaces
- **Occupied Slots**: Number of spaces with vehicles
- **Occupancy Rate**: Percentage of occupied spaces
- **Per-frame Statistics**: Complete temporal analysis

Statistics are exported to JSON for further analysis and visualization.

## Project Structure
```
carparking/
├── carparking.py              # Main detection script
├── parking_slots.json         # Parking slot coordinates
├── DJI_0012.MOV              # Input video
├── parking_mask.png          # Parking lot mask
├── parking_output/           # Output directory
│   ├── output.mp4           # Processed video
│   ├── occupancy_stats.json # Statistics
│   └── loaded_slots_preview.jpg
└── README.md                 # This file
```

## Limitations and Future Work

### Current Limitations
- Requires pre-annotated parking slot coordinates
- Performance depends on camera angle and lighting
- May struggle with severe shadows or extreme lighting conditions

### Future Improvements
- Automatic parking slot detection using line detection
- Deep learning integration for improved accuracy
- Support for multiple camera angles
- Real-time API for parking availability
- Mobile app integration

## Technical Requirements

- **Python**: 3.7 or higher
- **OpenCV**: 4.5.0 or higher
- **NumPy**: 1.19.0 or higher
- **Storage**: ~2GB for video files
- **RAM**: 8GB recommended

## License

This project is developed for educational purposes as part of a computer vision course assignment.

## Acknowledgments

- Drone footage for dataset creation
- OpenCV community for excellent documentation
- Computer vision research community for foundational techniques

## Contact

For questions or feedback about this project, please open an issue in the repository.

---

**Note**: This is an academic project demonstrating computer vision techniques for real-world applications.