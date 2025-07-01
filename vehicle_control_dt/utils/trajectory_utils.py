import pickle
import os
import numpy as np
import torch

def load_trajectories(path):
    """Pickleファイルから軌跡を読み込み"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data  # list of dicts


def save_trajectory(trajectories, path):
    """
    軌跡データ（リストの辞書）を pickle ファイルに保存します。
    trajectories: List[Dict] 各要素は1エピソードの軌跡
    path: 保存先ファイルパス
    """
    with open(path, 'wb') as f:
        pickle.dump(trajectories, f)


def load_trajectory(path):
    """
    pickle ファイルから軌跡データを読み込みます。
    path: 読み込みファイルパス
    return: List[Dict] 各辞書に 'observations', 'actions', 'rewards', 'terminals' など
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def denormalize(x, mean, std):
    return x * std + mean


def yaw_to_sin_cos(yaw_rad):
    """
    ラジアン単位のyawをsin, cosに変換
    引数:
        yaw_rad: float or np.ndarray (ラジアン, 1次元 or 多次元)
    戻り値:
        sin_yaw, cos_yaw: np.ndarray
    """
    sin_yaw = np.sin(yaw_rad)
    cos_yaw = np.cos(yaw_rad)
    return sin_yaw, cos_yaw

def sin_cos_to_yaw(sin_yaw, cos_yaw):
    """
    sin, cosからyaw（ラジアン）を復元
    引数:
        sin_yaw: float or np.ndarray
        cos_yaw: float or np.ndarray
    戻り値:
        yaw_rad: np.ndarray（-π〜π）
    """
    yaw_rad = np.arctan2(sin_yaw, cos_yaw)  # -π 〜 π に自動で調整される
    return yaw_rad
