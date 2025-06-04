import os
import glob
from stable_baselines3.common.callbacks import CheckpointCallback

from typing import Dict, Any, Optional, List

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, HParam, Figure, Video, Image
from stable_baselines3.common.vec_env import VecEnv

class SaveOnlyRecentCallback(CheckpointCallback):
    def __init__(self,
                 save_freq: int,
                 save_path: str,
                 name_prefix: str = "recent_model",
                 verbose: int = 0,
                 keep_n_checkpoints: int = 1,
                 save_replay_buffer=False,
                 save_vecnormalize=False): # 保持する最新のチェックポイントの数
        """
        Callback for saving a model every `save_freq` calls
        and keeping only the `keep_n_checkpoints` most recent ones.

        :param save_freq: Save checkpoints every `save_freq` call.
        :param save_path: Path to the folder where the model will be saved.
        :param name_prefix: Common prefix to use for saving the model
        :param verbose: Verbosity level.
        :param keep_n_checkpoints: Number of recent checkpoints to keep. Older ones are deleted.
                                   Must be greater than 0.
        """
        # We explicitly set save_replay_buffer and save_vecnormalize to False
        # because the deletion logic here is simplified for only the main model .zip files,
        # matching the user's original CheckpointCallback setup.
        # If these were True, one would need to also delete corresponding
        # _replay_buffer.pkl and _vecnormalize.pkl files for each deleted model.
        super().__init__(save_freq=save_freq,
                         save_path=save_path,
                         name_prefix=name_prefix,
                         save_replay_buffer=False, # ユーザーの設定に基づきFalseに固定
                         save_vecnormalize=False, # ユーザーの設定に基づきFalseに固定
                         verbose=verbose)
        if keep_n_checkpoints <= 0:
            raise ValueError("keep_n_checkpoints must be greater than 0.")
        self.keep_n_checkpoints = keep_n_checkpoints

    def _on_step(self) -> bool:
        # 親クラスの_on_step()を呼び出し、モデルの保存などを実行させる
        # 親クラスがFalseを返した場合（例: 他のロジックで学習停止が決定された場合）、それに従う
        continue_training = super()._on_step()
        if not continue_training:
            return False

        # 親クラスの_on_step()でモデルが保存された場合のみ、古いチェックポイントの削除処理を行う
        # (self.n_calls % self.save_freq == 0 のタイミング)
        if self.n_calls % self.save_freq == 0:
            # 保存されたモデルファイルのパターン ({name_prefix}_{num_timesteps}_steps.zip)
            # 例: "recent_model_5000_steps.zip", "recent_model_10000_steps.zip"
            pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
            list_of_files = glob.glob(pattern)

            if not list_of_files:
                return True # 削除対象のファイルなし

            # ファイル名からタイムステップ数を抽出し、降順（新しいものが先）にソートする
            def extract_step_from_filename(filename):
                try:
                    # ファイル名 (例: "recent_model_10000_steps.zip") のベース名を取得し、分割
                    # -> "recent_model_10000_steps" -> parts: ["recent", "model", "10000", "steps"] (name_prefixによる)
                    # もしくは "recent_model_10000_steps.zip" -> name_parts: ["recent_model", "10000", "steps.zip"] (name_prefixがアンダースコアなしの場合)
                    # より堅牢な方法として、末尾から探す
                    basename = os.path.basename(filename) # "recent_model_10000_steps.zip"
                    # ".zip" を除去 -> "recent_model_10000_steps"
                    # "_steps" を除去 -> "recent_model_10000"
                    # "_" で分割し最後の要素 -> "10000"
                    stem = basename[:-len(".zip")] # "recent_model_10000_steps"
                    if stem.endswith("_steps"):
                        step_str = stem[:-len("_steps")].split('_')[-1]
                        return int(step_str)
                except (ValueError, IndexError) as e:
                    if self.verbose > 1:
                        print(f"Could not parse step number from {filename}: {e}")
                return -1 # パースできないファイルはソートで先頭に来ないようにする

            list_of_files.sort(key=extract_step_from_filename, reverse=True) # 新しい順

            # 保持する数を超える古いファイルを削除
            files_to_delete = list_of_files[self.keep_n_checkpoints:]

            for file_to_delete in files_to_delete:
                try:
                    os.remove(file_to_delete)
                    if self.verbose > 1:
                        print(f"Removed old model checkpoint: {file_to_delete}")
                except OSError as e:
                    # エラーが発生しても学習は継続
                    if self.verbose > 0:
                        print(f"Error removing old model checkpoint {file_to_delete}: {e}")
        return True



class HyperParametersWriteCallback(BaseCallback):
    """
    コールバッククラス。
    トレーニング開始時にモデルのハイパーパラメータや指定された設定情報をTensorBoardに記録します。
    オプションで、トレーニング中に現在の学習率などの動的パラメータをTensorBoardに記録します。
    """
    def __init__(self,
                 config_data: Optional[Dict[str, Any]] = None,
                 hparams_to_log_on_start: Optional[Dict[str, Any]] = None,
                 log_model_architecture: bool = False,
                 log_model_specific_params: bool = True,
                 verbose: int = 0):
        """
        :param config_data: トレーニング開始時にテキストとして記録する一般的な設定情報 (辞書型)。
        :param hparams_to_log_on_start: トレーニング開始時にTensorBoardのHParamsタブに記録する追加のハイパーパラメータの辞書。
                                       Noneの場合、モデルの主要なパラメータを自動で抽出しようと試みます。
        :param log_model_architecture: Trueの場合、モデルのポリシーネットワーク構造をテキストとして記録しようと試みます。
                                        (torch.nn.Module の __repr__ を利用)
        :param log_model_specific_params: Trueの場合、トレーニング開始時にモデル特有のパラメータ (`get_parameters()`) を記録します。
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.config_data = config_data if config_data is not None else {}
        self.hparams_to_log_on_start = hparams_to_log_on_start if hparams_to_log_on_start is not None else {}
        self.log_model_architecture = log_model_architecture
        self.log_model_specific_params = log_model_specific_params
        
        # SummaryWriterのインスタンス。_on_training_startで初期化。
        self.writer = None 
        # PyTorchのSummaryWriterを直接使うため、loggerとは別に管理。

    def _get_tensorboard_writer(self):
        """
        Stable Baselines3のLoggerからTensorBoardのSummaryWriterインスタンスを取得します。
        """
        if self.writer is not None:
            return self.writer

        # self.logger (Loggerオブジェクト) が存在するか確認
        if self.logger is None:
            if self.verbose > 0:
                print("Warning: self.logger is None. Cannot get TensorBoard writer.")
            return None
            
        # self.logger.output_formats リストから TensorBoardOutputFormat を探す
        tb_formatter = None
        for formatter in self.logger.output_formats:
            if "TensorBoardOutputFormat" in type(formatter).__name__: # クラス名で判定
                tb_formatter = formatter
                break
        
        if tb_formatter is not None and hasattr(tb_formatter, "writer"):
            self.writer = tb_formatter.writer
            return self.writer
        else:
            if self.verbose > 0:
                print("Warning: TensorBoardOutputFormat not found in logger or writer not available. Cannot log custom data to TensorBoard.")
            return None

    def _on_training_start(self) -> None:
        """
        トレーニング開始時に呼び出されます。
        設定情報とハイパーパラメータをTensorBoardに記録します。
        """
        writer = self._get_tensorboard_writer()
        if writer is None:
            return # writer がなければ何もできない

        # 1. 指定された一般的な設定情報 (config_data) をテキストとして記録
        if self.config_data:
            config_str_parts = [f"- {key}: {value}" for key, value in self.config_data.items()]
            # Markdown形式で整形して記録
            config_md = "## General Configuration\n" + "\n".join(config_str_parts)
            writer.add_text("Setup/1_General_Configuration", config_md, global_step=0)
            if self.verbose > 0:
                print("Logged general configuration to TensorBoard.")

        # 2. モデルの主要なハイパーパラメータをHPARAMSとテキストで記録
        hparams_dict_for_hparam_tab = {} 
        hparams_text_parts = []

        # ユーザーが明示的に指定したハイパーパラメータを追加
        if self.hparams_to_log_on_start:
            for key, value in self.hparams_to_log_on_start.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    hparams_dict_for_hparam_tab[f"user/{key}"] = value
                hparams_text_parts.append(f"- user/{key}: {value}")


        # モデル固有のパラメータを `get_parameters()` から取得 (有効な場合)
        if self.log_model_specific_params and hasattr(self.model, "get_parameters"):
            try:
                model_params = self.model.get_parameters()
                hparams_text_parts.append("\n### Model Internal Parameters (`get_parameters()`):")
                for key, value in model_params.items():
                    param_tag = f"model_internal/{key}"
                    if isinstance(value, (int, float, str, bool, type(None))):
                        hparams_dict_for_hparam_tab[param_tag] = value
                        hparams_text_parts.append(f"- {key}: {value}")
                    elif isinstance(value, dict): # policy_kwargs など
                        hparams_text_parts.append(f"- {key}:")
                        for sub_key, sub_value in value.items():
                            sub_param_tag = f"{param_tag}/{sub_key}"
                            if isinstance(sub_value, (int, float, str, bool, type(None))):
                                hparams_dict_for_hparam_tab[sub_param_tag] = sub_value
                                hparams_text_parts.append(f"  - {sub_key}: {sub_value}")
                            else:
                                hparams_text_parts.append(f"  - {sub_key}: {str(sub_value)}") # 文字列として記録
                    else:
                        hparams_text_parts.append(f"- {key}: {str(value)}") # 文字列として記録
                if self.verbose > 0:
                    print(f"Extracted model internal parameters for HParams: {list(model_params.keys())}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Could not log model internal parameters via get_parameters(): {e}")
                hparams_text_parts.append(f"\n_Could not retrieve model_internal parameters: {e}_")

        # 主要なモデル属性からハイパーパラメータを取得
        hparams_text_parts.append("\n### Model Core Attributes:")
        
        # learning_rate_initial は、スケジューラオブジェクトそのものではなく、初期値を取得しようと試みる
        lr_initial_value = "N/A"
        if hasattr(self.model, "learning_rate"):
            if callable(self.model.learning_rate): # スケジューラの場合 (例: lambda p: 0.001 * p)
                try:
                    # progress=1.0 (開始時) または progress=0.0 (終了時)で評価。通常は1.0を使う。
                    lr_initial_value = self.model.learning_rate(1.0) 
                except Exception:
                    # スケジューラの初期値取得に失敗した場合、オブジェクトの文字列表現を記録
                    lr_initial_value = str(self.model.learning_rate)
            else: # 固定値の場合
                lr_initial_value = self.model.learning_rate
        
        env_id_str = "N/A"
        try:
            # VecEnvの場合
            if self.model.env and hasattr(self.model.env, "envs") and self.model.env.envs and \
               hasattr(self.model.env.envs[0], "spec") and self.model.env.envs[0].spec is not None:
                env_id_str = self.model.env.envs[0].spec.id
            # 単一環境の場合 (例: DummyVecEnv(lambda: gym.make(...)) の内部など)
            elif self.model.env and hasattr(self.model.env, "spec") and self.model.env.spec is not None:
                 env_id_str = self.model.env.spec.id
            # それでも取得できない場合 (gymnasium.make() で直接ラップされていない場合など)
            elif self.model.env and hasattr(self.model.env, "unwrapped") and hasattr(self.model.env.unwrapped, "spec") and self.model.env.unwrapped.spec is not None:
                env_id_str = self.model.env.unwrapped.spec.id

        except AttributeError:
            if self.verbose > 1:
                print("Debug: Could not determine environment ID automatically.")
            pass # env_id_str は "N/A" のまま

        core_attrs = {
            "gamma": getattr(self.model, "gamma", "N/A"),
            "learning_rate_initial": lr_initial_value,
            "n_steps": getattr(self.model, "n_steps", "N/A"), # PPO, A2C
            "batch_size": getattr(self.model, "batch_size", "N/A"), # PPO, SAC, TD3
            "policy_name": getattr(self.model, "_policy_aliases", {}).get(type(self.model.policy).__name__, type(self.model.policy).__name__),
            "environment_id": env_id_str
        }
        for key, value in core_attrs.items():
            param_tag = f"model_core/{key}"
            # HParamに記録できる型かチェック
            if isinstance(value, (int, float, str, bool)) or value is None:
                 hparams_dict_for_hparam_tab[param_tag] = value
            else: # それ以外の型は文字列に変換してHParamに記録（警告を出すことも検討）
                 hparams_dict_for_hparam_tab[param_tag] = str(value)
            hparams_text_parts.append(f"- {key}: {value}")
        
        # HPARAMSタブへの記録 (メトリクスはここでは記録しないので空辞書)
        if hparams_dict_for_hparam_tab:
            try:
                # HParamの値は int, float, str, bool, None である必要があるため最終サニタイズ
                # (既にある程度型チェックしているが念のため)
                sanitized_hparams = {}
                for k, v in hparams_dict_for_hparam_tab.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        sanitized_hparams[k] = v
                    else:
                        sanitized_hparams[k] = str(v) # TensorBoardが扱えるように文字列化
                        if self.verbose > 0:
                            print(f"Warning: HParam '{k}' with value '{v}' (type: {type(v)}) was converted to string for TensorBoard HParams tab.")
                
                writer.add_hparams(sanitized_hparams, {}, run_name=".", global_step=0) # run_name="." で現在のrunに紐づく
                if self.verbose > 0:
                    print(f"Logged HParams to TensorBoard: {list(sanitized_hparams.keys())}")
            except Exception as e:
                 if self.verbose > 0:
                    print(f"Warning: Error logging HParams to TensorBoard: {e}")
        
        # テキストとしてのハイパーパラメータ記録
        hparams_md = "## Hyperparameter Configuration\n" + "\n".join(hparams_text_parts)
        writer.add_text("Setup/2_Hyperparameters", hparams_md, global_step=0)
        if self.verbose > 0:
            print("Logged hyperparameter text to TensorBoard.")

        # モデルアーキテクチャの記録 (有効な場合)
        if self.log_model_architecture and hasattr(self.model, "policy") and self.model.policy is not None:
            try:
                # policyオブジェクトの文字列表現 (torch.nn.Moduleの__repr__) を利用
                policy_arch_str = str(self.model.policy)
                arch_md = f"## Model Architecture (`{type(self.model.policy).__name__}`)\n```\n{policy_arch_str}\n```"
                writer.add_text("Setup/3_Model_Architecture", arch_md, global_step=0)
                if self.verbose > 0:
                    print("Logged model architecture to TensorBoard.")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Could not log model architecture: {e}")
    
    def _on_step(self) -> bool:
        return True  # トレーニングを継続するためにTrueを返す