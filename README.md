# ID Card Extraction & Liveness API

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

A FastAPI service for extracting data from ID cards, verifying liveness, and comparing faces using webcam frames.

## Features

- ID card data extraction from webcam frames
- Liveness detection
- Face comparison between images
- Real-time processing status tracking

## Prerequisites

- Python 3.11
- Git LFS (for handling large model files)
- pip

## Installation

1. **Install Git LFS** (required for large model files):

```bash
# On Ubuntu/Debian
sudo apt-get install git-lfs

# On macOS (with Homebrew)
brew install git-lfs

# Initialize Git LFS
git lfs install
```

2. **Clone the repository** (with LFS files):

```bash
git clone <repository-url>
cd <repository-directory>
```

3. **Set up Python virtual environment**:

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the FastAPI server**:

```bash
uvicorn main:app --reload
```

2. **Access the API documentation**:

Open your browser to [http://localhost:8000/docs](http://localhost:8000/docs) for interactive OpenAPI documentation.

## API Endpoints

- `GET /health` - Health check
- `POST /id/process-frame/` - Process ID card frame
- `POST /id/reset-extraction/` - Reset extraction session
- `GET /id/extraction-status/` - Get extraction status
- `POST /image/compare_images` - Compare two images
- `POST /image/check_liveness` - Check liveness from image

## Configuration

For detailed configuration options, please check the `.env` file or environment variables documentation.

## License

Copyright (c) 2025 ITS All rights reserved.

This software is proprietary and confidential. Unauthorized copying, use,
modification, or distribution of this software is strictly prohibited unless
explicitly authorized in writing by ITS.

---

For more detailed API documentation and to try out the endpoints interactively, visit the `/docs` endpoint once the server is running.