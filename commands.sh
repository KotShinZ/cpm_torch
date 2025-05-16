poetry config pypi-token.pypi "キー" #pypi APIトークン
poetry add torch torchvision torchaudio --source torch_cu124 #torchのインストール
poetry add $( cat requirements.txt ) #requirements.txtのインストール
poetry install #poetryから各ライブラリをインストール
poetry publish --build #poetryでパッケージをビルドしてpublish