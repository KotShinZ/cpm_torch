FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# タイムゾーン
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# pyenvで使用するPythonバージョン（ビルド引数として指定可能）
ARG PYTHON_VERSION=3.9.17

# 必要なパッケージとPythonビルド依存関係のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
  # 既存のパッケージ
  git \
  curl \
  libopencv-dev \
  ffmpeg \
  # pyenv と Python ビルドに必要な依存関係
  make \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  wget \
  llvm \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libxml2-dev \
  libxmlsec1-dev \
  libffi-dev \
  liblzma-dev \
  python3-venv \
  # aptキャッシュのクリーンアップ
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# pyenvの環境変数を設定
ENV PYENV_ROOT /root/.pyenv
# Poetryの実行ファイルパス ($HOME/.local/bin) もPATHに追加
ENV PATH $PYENV_ROOT/bin:$PYENV_ROOT/shims:/root/.local/bin:$PATH

# pyenvのインストール
RUN curl https://pyenv.run | bash

# .bashrcにpyenvの初期化コマンドを追加
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# 指定されたPythonバージョンをインストールし、グローバルに設定
RUN CFLAGS="-O2 -g0" PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    # pipとsetuptoolsをアップグレードし、wheelをインストール
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Poetryのインストール (公式推奨のインストーラースクリプトを使用)
    curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2 --yes && \
    # Poetryコマンドをフルパスで実行して設定を行う
    # Poetry 1.2+ では --global オプションは不要 (デフォルトでグローバル設定)
    /root/.local/bin/poetry config virtualenvs.create false && \
    /root/.local/bin/poetry config virtualenvs.in-project false && \
    # Pythonインストール後のクリーンアップ
    rm -rf $PYENV_ROOT/versions/${PYTHON_VERSION}/share/doc && \
    rm -rf $PYENV_ROOT/versions/${PYTHON_VERSION}/share/man && \
    find $PYENV_ROOT/versions/${PYTHON_VERSION} -name '*.pyc' -delete && \
    find $PYENV_ROOT/versions/${PYTHON_VERSION} -name '__pycache__' -type d -exec rm -r {} +

# Pythonのバージョン、pipのバージョン、Poetryのバージョン、それぞれのパスを確認
RUN python --version && \
    pip --version && \
    poetry --version && \
    which python && \
    which pip && \
    which poetry

# Poetryのグローバル設定確認 (オプション)
# --list オプションは引き続き利用可能
RUN poetry config --list

# 作業ディレクトリの設定 (任意)
# WORKDIR /app

# Poetryのプロジェクトファイルがあればコピーして依存関係をインストールする例
# COPY pyproject.toml poetry.lock* ./
# RUN poetry install --no-interaction --no-ansi --no-root --no-dev

# ポート開放やCMD/ENTRYPOINTはプロジェクトに応じて設定
# EXPOSE 8000
# CMD ["python", "your_script.py"]