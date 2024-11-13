from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import queue
import threading
from dataclasses import dataclass
from typing import Dict, Any
import cv2

@dataclass
class CameraProperties:
    width: int = 640
    height: int = 480
    fps: int = 30

class BrowserVideoCapture:
    # Define constants to match OpenCV's property IDs
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self, device_index=0):
        """
        Initialize the BrowserVideoCapture instance.
        
        Args:
            device_index (int): Camera device index (used for key generation)
        """
        self.stream_key = f"camera_{device_index}"
        
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep the latest frame
        self.is_running = False
        self._lock = threading.Lock()
        
        # Store camera properties
        self._properties = CameraProperties()
        
        # Initialize the WebRTC streamer with video transform
        self.webrtc_ctx = webrtc_streamer(
            key=self.stream_key,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_frame_callback=self._video_frame_callback,
            async_processing=True,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": self._properties.width},
                    "height": {"ideal": self._properties.height},
                    "frameRate": {"ideal": self._properties.fps}
                }
            }
        )
        
        # Update running state based on WebRTC context
        if self.webrtc_ctx.state.playing:
            self.is_running = True

    def _video_frame_callback(self, frame):
        """
        Callback function to handle incoming video frames.
        """
        try:
            # Convert VideoFrame to numpy array using RGB format
            img = frame.to_ndarray(format="rgb24")
            
            # Convert RGB to BGR for OpenCV compatibility
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Update actual frame dimensions
            self._properties.width = img.shape[1]
            self._properties.height = img.shape[0]
            
            # Always keep only the latest frame
            if self.frame_queue.full():
                self.frame_queue.get_nowait()  # Remove old frame
            self.frame_queue.put_nowait(img)
        except Exception as e:
            print(f"Error in frame callback: {str(e)}")
        return frame

    def get(self, prop_id: int) -> float:
        """Get a camera property value."""
        with self._lock:
            if prop_id == self.CAP_PROP_FRAME_WIDTH:
                return float(self._properties.width)
            elif prop_id == self.CAP_PROP_FRAME_HEIGHT:
                return float(self._properties.height)
            elif prop_id == self.CAP_PROP_FPS:
                return float(self._properties.fps)
            return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        """Set a camera property value."""
        with self._lock:
            try:
                if prop_id == self.CAP_PROP_FRAME_WIDTH:
                    self._properties.width = int(value)
                elif prop_id == self.CAP_PROP_FRAME_HEIGHT:
                    self._properties.height = int(value)
                elif prop_id == self.CAP_PROP_FPS:
                    self._properties.fps = int(value)
                else:
                    return False

                # Note: WebRTC constraints can't be updated after initialization
                # Changes will take effect on next stream start
                return True
            except Exception:
                return False

    def isOpened(self):
        """Check if the video stream is open and running."""
        with self._lock:
            self.is_running = self.webrtc_ctx.state.playing
            return self.is_running

    def read(self):
        """Read a frame from the video stream."""
        if not self.isOpened():
            return False, None
        
        try:
            frame = self.frame_queue.get(timeout=1.0)  # 1 second timeout
            return True, frame
        except queue.Empty:
            return False, None

    def release(self):
        """Release the video capture resources."""
        with self._lock:
            self.is_running = False
            # Clear the frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
