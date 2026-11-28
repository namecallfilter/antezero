import subprocess
import sys
import time
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("yt-dlp is not installed. Please install it using: pip install yt-dlp")
    sys.exit(1)


# Edit these values before running this script.
# Dictionary of {Filename: URL} to download sequentially.
# The .mp4 extension will be added automatically if omitted.
VIDEOS_TO_DOWNLOAD = {
    "bu_erratic_deck_11-11-26": "https://www.youtube.com/watch?v=0kX6s4QM-EE",
    "bu_anaglyph_deck_11-10-26": "https://www.youtube.com/watch?v=KAqHsU2t_f8",
    "bu_magic_deck_11-9-26": "https://www.youtube.com/watch?v=J-O0iYtULa4",
    "bu_erratic_deck_11-5-26": "https://www.youtube.com/watch?v=88TnVBTs8HE",
    "bu_anaglyph_deck_10-31-26": "https://www.youtube.com/watch?v=0DcMgXRUDj8",
    "bu_erratic_deck_10-29-26": "https://www.youtube.com/watch?v=OdMPYOtOlBE",
    "bu_magic_deck_11-6-26": "https://www.youtube.com/watch?v=Gn-QP_dwV9w",
    "bu_anaglyph_deck_11-7-26": "https://www.youtube.com/watch?v=XSKyU4GM8js",
}

# Download settings
DOWNLOAD_VIDEOS = False
DOWNLOAD_VIDEO = DOWNLOAD_VIDEOS  # Alias for backward compatibility
DELAY_BETWEEN_DOWNLOADS = (
    2.0  # Seconds to wait between downloads to avoid rate limiting
)

# Frame scraping settings (if needed)
DOWNLOAD_FRAME = True
FRAME_VIDEO_URL = "https://www.youtube.com/watch?v=FyVXc7DaeqE"
FRAME_TIMESTAMP = "00:19:55.000"
FRAME_FILENAME = "full_deck_blind.jpg"

SCRIPT_DIR = Path(__file__).parent
FRAME_OUTPUT_DIR = SCRIPT_DIR.parent / "assets" / "sample_inputs"
VIDEO_OUTPUT_DIR = SCRIPT_DIR.parent / "assets" / "sample_inputs" / "videos"


def get_1080p_stream(video_url: str) -> str:
    """Extracts the direct stream URL for the 1080p video format."""
    # Force 1080p height. yt-dlp will throw an error if it doesn't exist.
    ydl_opts = {
        "quiet": True,
        "format": "bestvideo[height=1080]/best[height=1080]",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(video_url, download=False)
        except yt_dlp.utils.DownloadError as e:
            if "Requested format is not available" in str(e):
                raise ValueError("1080p format is not available for this video.") from e
            raise ValueError(f"Failed to fetch video info: {e}") from e

        if "requested_formats" in info:
            for f in info["requested_formats"]:
                if f.get("vcodec") != "none":
                    return f["url"]
        elif "url" in info:
            return info["url"]

        raise ValueError("Could not find a valid stream URL in the extracted info.")


def download_video(video_url: str, output_path: Path) -> None:
    """Download the complete video and merge separate audio/video streams."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid duplicate extensions if already provided in output_path
    stem_path = (
        output_path.with_suffix("")
        if output_path.suffix.lower() == ".mp4"
        else output_path
    )
    outtmpl = f"{stem_path}.%(ext)s"

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
    }

    print(f"Downloading video to: {stem_path}.mp4")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Failed to download video: {e}") from e


def download_all_videos(
    videos: dict[str, str],
    output_dir: Path,
    delay_seconds: float = 2.0,
) -> None:
    """Sequentially downloads multiple videos to avoid rate limiting and network saturation."""
    if not videos:
        print("No videos specified in VIDEOS_TO_DOWNLOAD.")
        return

    total = len(videos)
    print(f"Starting sequential download of {total} video(s)...")

    for idx, (filename, url) in enumerate(videos.items(), start=1):
        print(f"\n[{idx}/{total}] Target: {filename} -> {url}")
        output_path = output_dir / filename
        try:
            download_video(url, output_path)
            print(f"[{idx}/{total}] Successfully processed '{filename}'")
        except Exception as e:
            print(
                f"[{idx}/{total}] Error downloading '{filename}': {e}", file=sys.stderr
            )

        # Sleep briefly between downloads to avoid YouTube rate limiting (HTTP 429)
        if idx < total and delay_seconds > 0:
            print(
                f"Waiting {delay_seconds}s before next download to avoid rate limits..."
            )
            time.sleep(delay_seconds)


def grab_frame(video_url: str, timestamp: str, output_path: Path) -> None:
    """Grabs a specific frame using the stream URL and ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching 1080p stream URL for: {video_url}")

    try:
        stream_url = get_1080p_stream(video_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting frame at {timestamp}...")

    # We use ffmpeg to quickly seek to the timestamp and dump 1 frame.
    # OpenCV's cv2.VideoCapture seeking is notoriously broken on remote HLS/DASH streams.
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-ss",
        timestamp,  # Fast seek (must be before -i)
        "-i",
        stream_url,  # Input stream URL
        "-vframes",
        "1",  # Only grab 1 frame
        "-q:v",
        "2",  # High quality JPEG
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully saved frame to: {output_path}")
    except FileNotFoundError:
        print("Error: 'ffmpeg' is not installed or not in your PATH.", file=sys.stderr)
        print("ffmpeg is required to extract frames from the stream.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if DOWNLOAD_VIDEOS or DOWNLOAD_VIDEO:
        download_all_videos(
            VIDEOS_TO_DOWNLOAD,
            VIDEO_OUTPUT_DIR,
            delay_seconds=DELAY_BETWEEN_DOWNLOADS,
        )

    if DOWNLOAD_FRAME:
        grab_frame(
            FRAME_VIDEO_URL,
            FRAME_TIMESTAMP,
            FRAME_OUTPUT_DIR / FRAME_FILENAME,
        )
