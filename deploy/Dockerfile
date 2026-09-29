# HuggingFace Spaces (Docker SDK) 部署镜像
#
# 注意: 本文件是给 `python run.py prepare-hf` 生成的部署目录用的,
# 在那个目录里 Dockerfile 和 requirements.txt 都在根目录 ——
# 所以下面用的是根路径, 直接在主仓库根目录 build 会失败(路径对不上)。
#
# 三个平台约束, 每一条不满足都会启动失败:
#   1. 容器必须监听 7860 端口（可用 app_port 改, 但 7860 是默认）
#   2. 容器以 uid 1000 的非 root 用户运行, 写不了 / 和 /root
#   3. 只读文件系统之外的写入要落在 $HOME 下
FROM python:3.12-slim

# 平台以 uid 1000 运行容器, 镜像里必须有对应用户, 否则 pip --user 装的包找不到
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 服务端强制离线: 不调任何 API、不需要任何凭据, 也就没有泄露面
    GRAPHRAG_OFFLINE=1 \
    GRAPHRAG_LLM=deepseek \
    GRAPHRAG_EMBED=onnx \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR $HOME/app

# 先装依赖再拷代码 —— 代码改动不会让依赖层缓存失效
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# 启动前自检: 缓存覆盖不足时在日志里明确报出来, 而不是等用户点了才发现没答案
CMD ["sh", "-c", "python scripts/verify_replay.py --split train --all || true; exec python scripts/serve.py"]
