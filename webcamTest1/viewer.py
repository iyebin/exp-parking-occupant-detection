"""Small MJPEG web viewer shared by preview.py and 02_record.py.

Open http://<pi-ip>:<port> (or http://localhost:<port> through VS Code port forwarding).
Resizing + JPEG encoding run in their own thread at a capped rate, so the caller only
hands over a frame reference.
"""
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "<pi-ip>"


class WebView:
    def __init__(self, port=8000, max_fps=5.0, width=960, guides=False):
        self.port, self.period, self.width, self.guides = port, 1.0 / max_fps, width, guides
        self.lock = threading.Lock()
        self.latest = None  # (bgr, text lines, colour)
        self.jpeg = None
        self.running = True
        view = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                    self.end_headers()
                    last = None
                    try:
                        while view.running:
                            j = view.jpeg
                            if j is not None and j is not last:
                                last = j
                                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                            time.sleep(0.02)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                elif self.path.startswith("/snapshot"):
                    j = view.jpeg or b""
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.end_headers()
                    self.wfile.write(j)
                else:
                    page = (b"<html><body style='margin:0;background:#111'>"
                            b"<img src='/stream' style='width:100%;height:auto'></body></html>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(page)

        ThreadingHTTPServer.allow_reuse_address = True
        self.srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        threading.Thread(target=self._render, daemon=True).start()
        print(f"view: open http://{local_ip()}:{port}  (or http://localhost:{port} via VS Code Ports)")

    def update(self, bgr, lines=(), colour=(255, 255, 255)):
        with self.lock:
            self.latest = (bgr, list(lines), colour)

    def _render(self):
        while self.running:
            t0 = time.perf_counter()
            with self.lock:
                item, self.latest = self.latest, None
            if item is not None:
                bgr, lines, colour = item
                sc = self.width / bgr.shape[1]
                img = cv2.resize(bgr, (self.width, int(bgr.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                H, W = img.shape[:2]
                if self.guides:  # centre cross + thirds, to line people up with the floor marks
                    g = (0, 255, 255)
                    cv2.line(img, (W // 2, 0), (W // 2, H), g, 1)
                    cv2.line(img, (0, H // 2), (W, H // 2), g, 1)
                    for k in (1, 2):
                        cv2.line(img, (W * k // 3, 0), (W * k // 3, H), (0, 160, 160), 1)
                        cv2.line(img, (0, H * k // 3), (W, H * k // 3), (0, 160, 160), 1)
                bar = 24 * max(1, len(lines))
                cv2.rectangle(img, (0, 0), (W, bar + 4), (0, 0, 0), -1)
                for i, t in enumerate(lines):
                    cv2.putText(img, t, (8, 18 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.jpeg = buf.tobytes()
            time.sleep(max(0.0, self.period - (time.perf_counter() - t0)))

    def stop(self):
        self.running = False
        self.srv.shutdown()
        self.srv.server_close()