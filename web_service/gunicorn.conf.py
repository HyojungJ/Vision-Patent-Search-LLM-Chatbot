"""gunicorn 설정 (Docker 배포용)."""
import os

bind = "0.0.0.0:8000"

# 임베딩 모델(BGE-m3-ko, 약 2GB)과 Chroma 클라이언트를 프로세스마다 메모리에 올리므로
# worker는 1개로 두고, 스레드로 동시 요청(스트리밍 응답 포함)을 처리합니다.
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
worker_class = "gthread"
threads = int(os.getenv("GUNICORN_THREADS", "8"))

# LLM 답변 스트리밍(/chat/api/stream/)은 수십 초 이상 걸릴 수 있음
timeout = int(os.getenv("GUNICORN_TIMEOUT", "300"))
graceful_timeout = 30
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
