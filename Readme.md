# Weeks 1-3 Image Processing Lab

## Overview
This repository contains the completed image processing assignment exploring raw image pixel data, color channels, downsampling, intensity transformations, blurring, and edge detection using OpenCV, NumPy, and Matplotlib.

## Photo Verification & Integrity
- **Photographer**: Created original photograph manually.
- **Card Content**: Photo contains a handwritten physical card with "HCCGI Image Lab" and my initials (MH).
- **Image Content**: Features daily objects (spray bottle, container, pen) with clear boundaries and shapes, without sensitive data or human faces.

## Environment Setup & Run Instructions

### Prerequisites
- Python 3.8+

### Setup
1. Clone your private repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

### Task 1: Image Properties
- **Width**: 1600
- **Height**: 1200
- **Channels**: 3
- **NumPy Shape**: [1200, 1600, 3]
- **Pixel Count**: 1920000
- **Estimated Uncompressed Memory**: 5760000 bytes (~8 bits/channel)
- **OpenCV Loaded Color Format**: BGR

*Explanation of Dimensions*: NumPy arrays represent images in `(Height, Width, Channels)` format, where Height corresponds to row count (1200), Width to column count (1600), and Channels represent color depth (Blue, Green, Red).   