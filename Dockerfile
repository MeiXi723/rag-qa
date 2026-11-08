FROM python:3.11-slim

WORKDIR /app

# INSTALL_RERANK=false 可跳过 CrossEncoder（省 ~1GB torch，检索自动降级 RRF）
ARG INSTALL_RERANK=true
COPY requirements.txt requirements-rerank.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_RERANK" = "true" ]; then pip install --no-cache-dir -r requirements-rerank.txt; fi

COPY . .

RUN mkdir -p data/uploads data/chroma

EXPOSE 8000

# 存活探针走 /health（无鉴权零依赖）；start-period 覆盖 embedding 模型首载
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
