FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# タイムゾーン
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  curl \
  libopencv-dev \
  ffmpeg 
  #&& rm -rf /var/lib/apt/lists/*