import datetime
import cv2  # 画像処理ライブラリ OpenCV
import os  # オペレーティングシステム機能（ファイルパス操作など）
import io  # バイトストリーム操作
import PIL.Image, PIL.ImageDraw, PIL.ImageColor  # 画像処理ライブラリ Pillow
import base64  # Base64エンコーディング
import zipfile  # ZIPファイル操作
import json  # JSONデータ操作
import requests  # HTTPリクエスト
import matplotlib.pylab as plt  # グラフ描画ライブラリ Matplotlib
import glob  # ファイルパスのパターンマッチング
import torch
import numpy as np  # 数値計算ライブラリ NumPy

# IPython環境（Jupyterなど）が利用可能かチェックし、画像表示用の関数をインポート
try:
    from IPython.display import Image, HTML, clear_output, display

    ipython_available = True
except ImportError:
    ipython_available = False
    print("IPythonが見つかりません。画像表示関数は動作しません。")

# moviepy（動画編集ライブラリ）が利用可能かチェック
try:
    # os.environ['FFMPEG_BINARY'] = 'ffmpeg' # ffmpegのパス（必要に応じて設定）
    import moviepy.editor as mvp
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

    moviepy_available = True
except ImportError:
    moviepy_available = False
    print("moviepyが見つかりません。動画書き込み関数は動作しません。")
except OSError:
    moviepy_available = False
    print(
        "ffmpegが見つからないか、FFMPEG_BINARYのパスが正しくありません。動画書き込みが失敗する可能性があります。"
    )


def torch_to_numpy(tensor):
    """PyTorchテンソルをNumPy配列に変換する。"""
    # detach() で計算グラフから切り離し、cpu() でCPUに転送し、numpy() でNumPy配列に変換
    return tensor.detach().cpu().numpy()


def numpy_to_torch(array, device=None):
    """NumPy配列をPyTorchテンソルに変換する。"""
    tensor = torch.from_numpy(array)
    if device:
        tensor = tensor.to(device)  # 指定されたデバイス（CPU or GPU）に転送
    return tensor


def imread(url, max_size=None, mode=None):
    """URLまたはファイルパスから画像を読み込み、NumPy配列として返す。"""
    if url.startswith(("http:", "https:")):
        # URLの場合
        try:
            r = requests.get(url)
            r.raise_for_status()  # ステータスコードが異常なら例外を発生させる
            f = io.BytesIO(r.content)  # レスポンス内容をバイトストリームとして扱う
        except requests.exceptions.RequestException as e:
            print(f"画像URLからの取得エラー {url}: {e}")
            return None
    else:
        # ファイルパスの場合
        if not os.path.exists(url):
            print(f"エラー: ファイルが見つかりません {url}")
            return None
        f = url
    try:
        img = PIL.Image.open(f)  # Pillowで画像を開く
        if max_size is not None:
            # ANTIALIASは非推奨になったため、LANCZOSを使用
            img.thumbnail(
                (max_size, max_size), PIL.Image.Resampling.LANCZOS
            )  # 最大サイズにリサイズ
        if mode is not None:
            img = img.convert(mode)  # 指定されたモード（'RGB', 'L'など）に変換

        # NumPy配列に変換し、float32型で0-1の範囲に正規化
        img = np.array(img).astype(np.float32) / 255.0

        # グレースケールの場合、チャンネル数を3に拡張
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        # RGBA（4チャンネル）の場合、白色背景にアルファブレンディングしてRGB（3チャンネル）に変換
        elif img.ndim == 3 and img.shape[-1] == 4:
            alpha = img[..., 3:4]  # アルファチャンネルを抽出
            img = img[..., :3] * alpha + (
                1.0 - alpha
            )  # RGBチャンネルと背景色（白）をブレンド

        return img
    except Exception as e:
        print(f"画像の読み込みまたは処理エラー {url}: {e}")
        return None


def np2pil(a):
    """NumPy配列をPillow (PIL) Imageオブジェクトに変換する。"""
    if a.dtype in [np.float32, np.float64]:
        # float型の場合は0-1の範囲にクリップし、0-255のuint8型に変換
        a = np.uint8(np.clip(a, 0, 1) * 255)

    # 配列の次元数とチャンネル数に応じて適切なモードでPIL Imageを作成
    if a.ndim == 2:
        return PIL.Image.fromarray(a, mode="L")  # グレースケール
    elif a.ndim == 3 and a.shape[-1] == 3:
        return PIL.Image.fromarray(a, mode="RGB")  # RGB
    elif a.ndim == 3 and a.shape[-1] == 4:
        return PIL.Image.fromarray(a, mode="RGBA")  # RGBA
    else:
        raise ValueError(f"PIL変換でサポートされていないNumPy配列の形状: {a.shape}")


def imwrite(f, a, fmt=None):
    """画像データ（NumPy配列またはPyTorchテンソル）をファイルに書き込む。"""
    # 入力がPyTorchテンソルの場合、NumPy配列に変換
    if isinstance(a, torch.Tensor):
        a = torch_to_numpy(a)
    a = np.asarray(a)  # NumPy配列であることを保証

    pil_img = np2pil(a)  # NumPy配列をPIL Imageに変換

    if isinstance(f, str):
        # fがファイルパス（文字列）の場合
        if fmt is None:
            # フォーマットが指定されていない場合は拡張子から推定
            fmt = f.rsplit(".", 1)[-1].lower()
            if fmt == "jpg":
                fmt = "jpeg"
        try:
            # パスにディレクトリが含まれる場合、存在しなければ作成
            if "/" in f or "\\" in f:
                os.makedirs(os.path.dirname(f), exist_ok=True)
            # ファイルをバイナリ書き込みモードで開いて保存
            with open(f, "wb") as fp:
                pil_img.save(fp, fmt, quality=95)  # quality=95で保存
        except Exception as e:
            print(f"画像ファイルへの書き込みエラー {f}: {e}")
    elif hasattr(f, "write"):
        # fがファイルライクオブジェクト（BytesIOなど）の場合
        pil_img.save(
            f, fmt if fmt else "png", quality=95
        )  # フォーマット指定がなければPNGで保存
    else:
        raise TypeError(f"引数 'f' にサポートされていない型: {type(f)}")


def imencode(a, fmt="jpeg"):
    """画像データ（NumPy配列またはPyTorchテンソル）を指定されたフォーマットでエンコードし、バイト列を返す。"""
    # 入力がPyTorchテンソルの場合、NumPy配列に変換
    if isinstance(a, torch.Tensor):
        a = torch_to_numpy(a)
    a = np.asarray(a)  # NumPy配列であることを保証

    # チャンネル数が4（RGBA）の場合、PNGフォーマットを使用（JPEGはRGBA非対応）
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = "png"

    f = io.BytesIO()  # メモリ上のバイトストリーム
    imwrite(f, a, fmt)  # imwriteを使ってバイトストリームに書き込む
    return f.getvalue()  # バイト列を取得して返す


def im2url(a, fmt="jpeg"):
    """画像データ（NumPy配列またはPyTorchテンソル）をData URL形式に変換する。"""
    encoded = imencode(a, fmt)  # 画像をエンコードしてバイト列を取得
    base64_byte_string = base64.b64encode(encoded).decode("ascii")  # Base64エンコード
    # Data URL形式の文字列を作成して返す
    return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


def imshow(a, target_width=256, target_height=256, fmt="jpeg"):
    """
    画像（NumPy配列またはPyTorchテンソル）を指定解像度にリサイズして表示する（IPython環境用）。
    """
    if not ipython_available:
        print("imshow関数はIPython.displayが必要です。")
        # 代替案: matplotlibで表示（コメントアウト）
        # plt.imshow(torch_to_numpy(a) if isinstance(a, torch.Tensor) else a)
        # plt.show()
        return

    try:
        # 入力がPyTorchテンソルの場合、NumPy配列に変換
        if isinstance(a, torch.Tensor):
            a = torch_to_numpy(a)

        # 入力がNumPy配列であることを確認
        if not isinstance(a, np.ndarray):
            raise TypeError(
                f"入力 'a' はNumPy配列またはPyTorchテンソルである必要がありますが、{type(a)} を受け取りました。"
            )

        # グレースケール（2次元配列）や単一チャンネルの場合、表示用にRGBに変換
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)  # 3チャンネルにスタック
        elif a.ndim == 3 and a.shape[-1] == 1:
            a = np.concatenate([a] * 3, axis=-1)  # 3チャンネルに結合

        # float型で範囲が[0, 1]外の場合、正規化
        if a.dtype in [np.float32, np.float64]:
            if a.min() < 0 or a.max() > 1:
                a = (a - a.min()) / (
                    a.max() - a.min() + 1e-6
                )  # [0, 1]に正規化（ゼロ除算防止のためepsilon追加）
            a = (a * 255).astype(np.uint8)  # [0, 255]のuint8型に変換
        elif a.dtype != np.uint8:
            a = a.astype(np.uint8)  # uint8型に変換

        # JPEG形式でエンコードする場合、入力がRGBAならRGBに変換
        if fmt == "jpeg" and a.shape[-1] == 4:
            # 白色背景にアルファブレンディング
            alpha = a[..., 3:4].astype(np.float32) / 255.0
            a = (a[..., :3] * alpha + 255 * (1.0 - alpha)).astype(np.uint8)
        elif a.shape[-1] == 1:  # グレースケールをRGBに変換
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)

        # OpenCVは HxWxC フォーマットを期待
        if a.shape[-1] not in [3, 4]:
            raise ValueError(
                f"入力配列 'a' は3 (RGB) または 4 (RGBA) チャンネルを持つ必要がありますが、形状 {a.shape} を受け取りました。"
            )

        # OpenCVを使ってリサイズ
        target_width = int(target_width)
        target_height = int(target_height)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("目標の幅と高さは正の整数である必要があります。")

        # 縮小か拡大かに応じて補間方法を選択
        current_height, current_width = a.shape[:2]
        if target_width < current_width or target_height < current_height:
            interpolation = cv2.INTER_AREA  # 縮小時
        else:
            interpolation = cv2.INTER_LINEAR  # 拡大時（またはcv2.INTER_CUBIC）

        resized_a = cv2.resize(
            a, (target_width, target_height), interpolation=interpolation
        )

        # リサイズされた画像をエンコード
        fmt_cv = f".{fmt}"  # OpenCV用のフォーマット文字列（例: '.jpeg'）
        is_success, buffer = cv2.imencode(
            fmt_cv, resized_a
        )  # 画像を指定フォーマットでメモリ上のバッファにエンコード

        if not is_success:
            raise ValueError(f"画像を {fmt} 形式にエンコードできませんでした。")

        # IPython.display.Image を使って表示
        display(Image(data=buffer.tobytes()))  # バッファのバイトデータを渡す

    except Exception as e:
        print(f"imshow関数でエラーが発生しました: {e}")
        # 必要であればトレースバックを表示:
        # import traceback
        # traceback.print_exc()


def tile2d(a, w=None):
    """複数の画像をタイル状に並べて1つの画像にする。"""
    # 入力がPyTorchテンソルの場合、NumPy配列に変換
    if isinstance(a, torch.Tensor):
        a = torch_to_numpy(a)
    a = np.asarray(a)  # NumPy配列であることを保証

    if w is None:
        # タイルの幅が指定されていない場合、入力画像数から平方根で推定
        w = int(np.ceil(np.sqrt(len(a))))

    # 各タイルの高さ(th)と幅(tw)を取得
    th, tw = a.shape[1:3]

    # タイル数がwの倍数になるように、不足分をパディングで埋める
    pad = (w - len(a)) % w
    # NumPyのpad関数用にパディング設定を作成: [(0軸の上側, 0軸の下側), (1軸の上側, 1軸の下側), ...]
    padding_config = [(0, pad)] + [(0, 0)] * (a.ndim - 1)
    a = np.pad(a, padding_config, "constant", constant_values=0)  # 定数値0でパディング

    h = len(a) // w  # タイルの高さ（行数）
    # (タイル数, th, tw, ...) -> (h, w, th, tw, ...) にリシェイプ
    a = a.reshape([h, w] + list(a.shape[1:]))

    # np.rollaxis(a, 2, 1) と同等の操作を np.transpose で行う
    # 軸を (h, w, th, tw, ...) から (h, th, w, tw, ...) に入れ替える
    perm = [0, 2, 1] + list(range(3, a.ndim))
    a = np.transpose(a, perm)

    # (h, th, w, tw, ...) -> (h*th, w*tw, ...) にリシェイプして最終的なタイル画像を作成
    a = a.reshape([th * h, tw * w] + list(a.shape[4:]))
    return a  # NumPy配列として返す


def zoom(img, scale=4):
    """画像を単純な繰り返しで拡大（最近傍補間）する。"""
    # 入力がPyTorchテンソルの場合、NumPy配列に変換
    if isinstance(img, torch.Tensor):
        img = torch_to_numpy(img)
    img = np.asarray(img)  # NumPy配列であることを保証

    # 高さ方向（軸0）にscale回繰り返し
    img = np.repeat(img, scale, axis=0)
    # 幅方向（軸1）にscale回繰り返し
    img = np.repeat(img, scale, axis=1)
    return img  # NumPy配列として返す


# --- VideoWriter クラス (moviepyが必要) ---
if moviepy_available:

    class VideoWriter:
        """画像フレームを追加して動画ファイルを作成するクラス。"""

        def __init__(self, filename="_tmp.mp4", fps=30.0, **kw):
            """コンストラクタ。
            Args:
                filename (str): 出力動画ファイル名。
                fps (float): フレームレート。
                **kw: FFMPEG_VideoWriterに渡す追加パラメータ。
            """
            self.writer = None  # FFMPEG_VideoWriterのインスタンスを保持
            self.params = dict(
                filename=filename, fps=fps, **kw
            )  # パラメータを辞書に保存

        def add(self, img):
            """動画にフレームを追加する。"""
            # 入力がPyTorchテンソルの場合、NumPy配列に変換
            if isinstance(img, torch.Tensor):
                img = torch_to_numpy(img)
            img = np.asarray(img)  # NumPy配列であることを保証

            # float型 [0, 1] を uint8型 [0, 255] に変換
            if img.dtype in [np.float32, np.float64]:
                img = np.uint8(np.clip(img, 0, 1) * 255)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)  # uint8型に変換

            # ライターが未初期化の場合、最初のフレームからサイズを取得して初期化
            if self.writer is None:
                h, w = img.shape[:2]  # フレームの高さと幅を取得
                # グレースケールの場合、動画用にRGBに変換
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3 and img.shape[-1] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # 変換後の形状を再度取得
                h, w = img.shape[:2]

                # チャンネル数が3（RGB）でない場合の処理
                if img.shape[-1] != 3:
                    print(
                        f"警告: VideoWriterはRGB画像（3チャンネル）を想定していますが、形状 {img.shape} を受け取りました。変換を試みます。"
                    )
                    if img.shape[-1] == 4:  # RGBA -> RGB
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    else:
                        print(f"エラー: 形状 {img.shape} の画像をRGBに変換できません。")
                        return  # このフレームはスキップ

                # 最終的な幅と高さでライターを初期化
                h, w = img.shape[:2]
                self.writer = FFMPEG_VideoWriter(
                    filename=self.params["filename"],
                    size=(w, h),
                    fps=self.params["fps"],
                    **self.params,
                )

            # 初期化後、グレースケールや単一チャンネルのフレームが来た場合の処理
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, -1)  # 3チャンネルに複製
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, -1)  # 3チャンネルに複製

            # 書き込み前にRGBAならRGBに変換
            if img.shape[-1] == 4:  # RGBA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # 最終確認：3チャンネルのuint8画像のみを書き込む
            if img.ndim == 3 and img.shape[-1] == 3:
                self.writer.write_frame(img)
            else:
                print(f"予期しない形状のフレームをスキップします: {img.shape}")

        def close(self):
            """動画ファイルを閉じて書き込みを完了する。"""
            if self.writer:
                try:
                    self.writer.close()
                except Exception as e:
                    print(f"ビデオライターのクローズ中にエラー発生: {e}")
                self.writer = None  # ライターをリセット

        def __enter__(self):
            """`with`構文で使用するためのメソッド。"""
            return self

        def __exit__(self, *kw):
            """`with`構文の終了時に自動的にclose()を呼び出す。"""
            self.close()
            # 一時ファイルの場合、IPython環境なら表示を試みる
            if (
                ipython_available
                and self.params.get("filename") == "_tmp.mp4"
                and os.path.exists("_tmp.mp4")
            ):
                try:
                    self.show()
                except Exception as e:
                    print(f"動画の表示に失敗しました: {e}")
                # 表示に失敗しても一時ファイルは削除
                try:
                    os.remove("_tmp.mp4")
                except OSError:
                    pass  # ファイルが削除できなくても無視

        def show(self, **kw):
            """作成した動画をIPython環境で表示する。"""
            self.close()  # まずファイルを閉じる
            fn = self.params.get("filename")
            if not fn or not os.path.exists(fn):
                print(f"動画ファイルが見つかりません: {fn}")
                return
            if ipython_available:
                # 表示幅のデフォルト値を設定
                kw.setdefault("width", 400)
                try:
                    display(mvp.ipython_display(fn, **kw))
                except Exception as e:
                    print(f"動画 {fn} の表示中にエラー発生: {e}")
            else:
                print(f"動画 '{fn}' を表示できません。IPython環境ではありません。")

    class LoopWriter(VideoWriter):
        """動画の最初と最後をフェードで繋げてループ再生できるようにするクラス。"""

        def __init__(self, *a, **kw):
            self.fade_len_sec = kw.get("fade_len", 1.0)  # フェード時間（秒）を保存
            super().__init__(*a, **kw)  # 親クラスの初期化を呼び出す
            self._intro = []  # フェードイン用のフレームバッファ
            self._outro = []  # フェードアウト用のフレームバッファ
            # fpsが設定された後にフェード長をフレーム数で計算
            self.fade_len = int(self.fade_len_sec * self.params["fps"])

        def add(self, img):
            """ループ用のフレームを追加する。"""
            # テンソル -> NumPy変換とデータ型処理を早期に行う
            if isinstance(img, torch.Tensor):
                img = torch_to_numpy(img)
            img = np.asarray(img)
            if img.dtype in [np.float32, np.float64]:
                img = np.clip(img, 0, 1)  # float型は[0, 1]にクリップ

            # ライター未初期化で、introバッファが満杯でない場合
            if self.writer is None and len(self._intro) < self.fade_len:
                # サイズ決定のために一時的にuint8に変換
                img_uint8 = (
                    (img * 255).astype(np.uint8)
                    if img.dtype in [np.float32, np.float64]
                    else img.astype(np.uint8)
                )
                # ライター初期化のために親クラスのaddを呼び出す
                super().add(img_uint8)
                # ライターが初期化されたら、元のフレームをintroバッファに保存
                if self.writer:
                    self._intro.append(img)  # 元のfloat/numpyフレームを保存
                return  # バッファリングまたは初期化試行後に終了

            # 通常のバッファリングロジック
            if len(self._intro) < self.fade_len:
                self._intro.append(img)  # introバッファに追加
                return

            # outroバッファに追加
            self._outro.append(img)
            # outroバッファがフェード長を超えたら、最も古いフレームを書き出す
            if len(self._outro) > self.fade_len:
                frame_to_write = self._outro.pop(0)  # 最も古いフレームを取得
                # 書き込み直前にuint8に変換して親クラスのaddを呼び出す
                super().add(frame_to_write)

        def close(self):
            """ループ動画を完成させてファイルを閉じる。"""
            if not self.writer:
                print(
                    "警告: LoopWriterはフレームが追加/ライターが初期化されないまま閉じられようとしています。"
                )
                super().close()  # 親クラスのcloseを呼び出す
                return

            # intro/outroバッファのサイズが一致しない場合（通常は起こらないはず）
            if len(self._intro) != self.fade_len or len(self._outro) != self.fade_len:
                print(
                    f"警告: クローズ時にフェードバッファのサイズが一致しません。intro={len(self._intro)}, outro={len(self._outro)}, fade_len={self.fade_len}"
                )
                # 最小の長さに合わせて処理を試みる
                min_len = min(len(self._intro), len(self._outro))
                self._intro = self._intro[:min_len]
                self._outro = self._outro[:min_len]
                self.fade_len = min_len  # フェード長も合わせる

            # バッファが正しく満たされている場合にフェード処理を実行
            if len(self._intro) == self.fade_len and self.fade_len > 0:
                for t in np.linspace(
                    0, 1, self.fade_len
                ):  # フェード係数tを0から1まで変化させる
                    # バッファからフレームを取り出し、float32に変換
                    intro_frame = np.asarray(self._intro.pop(0), dtype=np.float32)
                    outro_frame = np.asarray(self._outro.pop(0), dtype=np.float32)
                    # 線形補間でフレームをブレンド
                    img = intro_frame * t + outro_frame * (1.0 - t)
                    # 書き込み直前にuint8に変換して親クラスのaddを呼び出す
                    super().add(img)

            # 残っているoutroフレーム（正常なら空のはず）を書き出す
            while self._outro:
                super().add(self._outro.pop(0))

            super().close()  # 親クラスのcloseを呼び出してファイルを最終化

else:
    # moviepyが利用できない場合のダミークラス定義
    class VideoWriter:
        def __init__(self, *args, **kwargs):
            print(
                "警告: moviepyがインストールされていないため、VideoWriterは動作しません。"
            )

        def add(self, img):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *kw):
            pass

        def show(self, **kw):
            print("moviepyがインストールされていないため、動画を表示できません。")

    class LoopWriter(VideoWriter):
        def __init__(self, *args, **kwargs):
            print(
                "警告: moviepyがインストールされていないため、LoopWriterは動作しません。"
            )


# === 色関連の関数 ===
def create_rgb_image_from_hue_tensor_pil(
    hue_tensor: torch.Tensor, hue_max_value: int = 255
) -> torch.Tensor:
    """
    2次元の色相(Hue)テンソルから、彩度(Saturation)と明度(Value)が最大のRGB画像テンソルを作成する。
    HSVからRGBへの変換にはPillowライブラリを使用する。

    Args:
        hue_tensor (torch.Tensor): (H, W)形状の整数色相テンソル。
        hue_max_value (int): hue_tensor内の最大値（例: 255）。正規化に使用。

    Returns:
        torch.Tensor: (H, W, 3)形状のuint8型RGB画像テンソル。入力と同じデバイス上にある。
    """
    assert hue_tensor.ndim == 2, "入力テンソルは2次元(H, W)である必要があります。"
    input_device = hue_tensor.device  # 入力テンソルのデバイスを保持

    # Pillowで処理するために、色相テンソルをCPU上のNumPy配列(uint8)に変換
    hue_numpy = torch_to_numpy(hue_tensor).astype(np.uint8)

    # 彩度(S)と明度(V)を最大値(255)で埋めたNumPy配列を作成
    h, w = hue_numpy.shape
    saturation_numpy = np.full_like(hue_numpy, 255, dtype=np.uint8)
    value_numpy = np.full_like(hue_numpy, 255, dtype=np.uint8)

    # H, S, VのチャンネルをスタックしてHSV形式のNumPy配列を作成 (形状: H, W, 3)
    hsv_numpy = np.stack([hue_numpy, saturation_numpy, value_numpy], axis=-1)

    # HSV形式のNumPy配列をPillow Imageオブジェクトに変換 (モード'HSV')
    hsv_pil = PIL.Image.fromarray(hsv_numpy, mode="HSV")

    # Pillow ImageオブジェクトをRGBモードに変換
    rgb_pil = hsv_pil.convert("RGB")

    # RGBモードのPillow ImageオブジェクトをNumPy配列に変換 (形状: H, W, 3, dtype: uint8)
    rgb_numpy = np.array(rgb_pil)

    # NumPy配列をPyTorchテンソルに変換し、元のデバイスに転送
    rgb_tensor = numpy_to_torch(rgb_numpy, device=input_device)

    return rgb_tensor


# === 色関数 使用例（コメントアウト） ===
# hue_input_tensor = torch.randint(0, 256, size=(64, 64), dtype=torch.int32, device=device)
# print("入力テンソルの形状:", hue_input_tensor.shape)
# print("入力テンソルのデータ型:", hue_input_tensor.dtype)
# rgb_image_tensor = create_rgb_image_from_hue_tensor_pil(hue_input_tensor, hue_max_value=255)
# print("出力RGBテンソルの形状:", rgb_image_tensor.shape)
# print("出力RGBテンソルのデータ型:", rgb_image_tensor.dtype)
# # 表示 (CPUに転送してNumPyに変換してからimshowへ)
# imshow(rgb_image_tensor.cpu())

# 色リスト (NumPy配列として作成し、必要なら後でテンソルに変換)
# ID 0 (背景) も扱えるようにサイズを +1 する
color_list_np = np.random.randint(0, 256, size=(256 + 1,), dtype=np.uint8)

def map_tensor_to_rgb(map_tensor):
    ids = map_tensor[:, :, 0].long() % len(
        color_list_np
    )  # IDをlong型に変換し、色リストの範囲内に収める
    
    color_list_torch = torch.from_numpy(color_list_np).to(ids.device)  # 色リストをIDのデバイスに転送
    hues = color_list_torch[ids]  # (H, W)
    # 色相テンソルからRGB画像テンソルを生成
    im_tensor = create_rgb_image_from_hue_tensor_pil(hues)

    # IDが0（背景）のピクセルを白(255, 255, 255)にする
    background_mask = ids == 0  # 背景ピクセルのマスク (True/False)
    white_color = torch.tensor([255, 255, 255], dtype=torch.uint8, device=im_tensor.device)
    im_tensor[background_mask] = white_color  # マスクされた部分を白で上書き
    
    return im_tensor  # RGB画像テンソルを返す

# === 可視化関数 ===
def imshow_map(map_tensor):
    """マップテンソルの細胞IDに基づいて色付けし、画像として表示する。"""
    im_tensor = map_tensor_to_rgb(map_tensor)  # ID画像を生成

    # imshow関数で表示（テンソル -> NumPy -> 表示処理はimshow内部で行われる）
    imshow(im_tensor)


def imshow_map_area(map_tensor, _max=100.0, _min=0.0):
    """マップテンソルのチャンネル1（面積/密度）を指定範囲で正規化し、グレースケール画像として表示する。"""
    area = map_tensor[:, :, 1]  # 面積/密度チャンネルを取得
    # 値を指定された最小値(_min)と最大値(_max)の範囲にクリップ（制限）する
    area = torch.clamp(area, min=_min, max=_max)
    # [0, 255]の範囲に正規化
    normalized_area = (
        (area - _min) * 255.0 / (_max - _min + 1e-9)
    )  # ゼロ除算防止のためepsilon追加
    # 表示用にuint8型に変換
    im_tensor_area = normalized_area.to(torch.uint8)
    # グレースケール画像としてimshowで表示（imshowは単一チャンネルを適切に処理）
    imshow(im_tensor_area)


def imshow_map_area_autoRange(map_tensor):
    """マップテンソルのチャンネル1（面積/密度）を自動範囲で正規化し、グレースケール画像として表示する。"""
    area = map_tensor[:, :]  # 面積/密度チャンネルを取得
    _max = torch.max(area)  # 最大値を取得
    _min = torch.min(area)  # 最小値を取得
    # [0, 255]の範囲に正規化
    if _max > _min:
        normalized_area = (area - _min) * 255.0 / (_max - _min)
    else:
        normalized_area = torch.zeros_like(
            area
        )  # 最大値と最小値が同じ場合はゼロ除算を避け、ゼロにする
    im_tensor_area = normalized_area.to(torch.uint8)  # uint8型に変換
    # グレースケール画像として表示
    imshow(im_tensor_area)

