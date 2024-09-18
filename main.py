import os

# Set the environment variable to increase packet read attempts
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

import argparse
import logging
from colorama import init, Fore, Style
from face_detect import FaceDetection
from face_classification import FaceClassification

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    return logger

def main():
    init(autoreset=True)  # Initialize colorama
    print(Fore.GREEN + Style.BRIGHT + "Advanced Face Recognition and Classification System\n")

    parser = argparse.ArgumentParser(description="Advanced Face Recognition and Classification System")
    parser.add_argument("--input", help="Input directory containing video files")
    parser.add_argument("--output", help="Output directory for processed data")
    args = parser.parse_args()

    input_dir = args.input if args.input else input("Enter the input directory containing video files (default: ./input_videos): ") or "./input_videos"
    output_dir = args.output if args.output else input("Enter the output directory for processed data (default: ./output_faces): ") or "./output_faces"

    if not os.path.isdir(input_dir):
        print(Fore.RED + f"Error: Input directory '{input_dir}' does not exist.")
        return

    logger = setup_logging()

    face_detector = FaceDetection(logger)
    face_detector.process_videos(input_dir)

    face_classifier = FaceClassification(logger)
    face_classifier.get_embeddings()
    face_classifier.cluster_faces(output_dir)
    face_classifier.generate_report(output_dir)

    print(Fore.BLUE + Style.BRIGHT + f"\nProcessing complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
