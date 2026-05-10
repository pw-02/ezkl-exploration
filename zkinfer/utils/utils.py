from uuid import uuid4
import socket

def get_ip():
    # Gets the primary IP address (not always public)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = f"127.0.0.1_{uuid4().hex[:8]}"  # Fallback to localhost with a unique suffix
    finally:
        s.close()
    return ip