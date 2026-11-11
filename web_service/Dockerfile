# 특허 검색 AI 챗봇 (Django) — gunicorn으로 실행, nginx 뒤에서 서비스 (../docker-compose.yml)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# CPU용 PyTorch를 먼저 설치 (기본 CUDA 휠보다 이미지가 훨씬 작음)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# 정적 파일 수집 → STATIC_ROOT(/app/staticfiles), nginx가 공유 볼륨으로 서빙
RUN DJANGO_SECRET_KEY=collectstatic-only python manage.py collectstatic --noinput

# 비root 사용자로 실행
RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/sessions /cache/huggingface \
    && chown -R appuser:appuser /app /cache
USER appuser

EXPOSE 8000

# 시작 시 마이그레이션 적용 후 gunicorn 실행
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn config.wsgi:application -c gunicorn.conf.py"]
