#!/usr/bin/env python3
"""Runner script to launch the Clinical Screening Web Studio."""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set expandable segments to avoid CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    parser = argparse.ArgumentParser(description="Launch Retinal Disease Clinical Screening Web Studio")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    import uvicorn

    print("=" * 70)
    print(" 👁️  RETINAL DISEASE CLINICAL SCREENING WEB STUDIO")
    print(" Department of Information Technology, Jadavpur University")
    print(f" Listening on http://localhost:{args.port} (or http://{args.host}:{args.port})")
    print("=" * 70)

    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
