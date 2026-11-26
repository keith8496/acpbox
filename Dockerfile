FROM python:3.11-slim

RUN groupadd --gid 1000 user \
 && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/user user

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:/home/user/.opencode/bin:${PATH}"
RUN apt-get update \
 && apt-get install -y --no-install-recommends bash curl ca-certificates nodejs npm \
 && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir . \
 && chown -R user:user /app \
 && mkdir -p /data/workspace \
 && chown user:user /data/workspace \
 && chmod +x /app/entrypoint.sh

EXPOSE 8080

USER user

#RUN curl -fsSL https://antigravity.google/cli/install.sh | bash \
# && which agy \
# && agy --version

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["acpbox"]
