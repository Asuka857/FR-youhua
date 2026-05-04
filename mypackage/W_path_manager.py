"""
路径配置模块
功能：统一管理所有文件路径，支持自定义输出目录

创建时间：2025-03-20
"""

import os
from pathlib import Path
from typing import Union


# =============================================================================
# 核心路径配置
# =============================================================================

# 项目根目录（自动检测）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 默认基础输出目录（当前项目目录下的results文件夹）
DEFAULT_BASE_DIR = PROJECT_ROOT / 'results'

# 用户可通过环境变量自定义
BASE_DIR = Path(os.getenv('WATERML_DIR', str(DEFAULT_BASE_DIR)))


# =============================================================================
# 路径管理类
# =============================================================================

class PathManager:
    """
    路径管理器
    功能：统一管理所有输入输出路径
    """

    def __init__(self, base_dir: Union[str, Path] = None):
        """
        初始化路径管理器

        参数：
            base_dir: 基础输出目录，默认使用BASE_DIR
        """
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 输入数据路径
    # =========================================================================

    def get_train_data_path(self, crop: str, readday: str, indicator: str) -> Path:
        """
        获取训练数据路径

        参数：
            crop: 作物类型
            readday: 数据日期
            indicator: 指标名称

        返回：
            训练数据文件路径
        """
        # 训练数据可能在原始位置，也可能在本地
        # 优先使用本地，如果不存在则使用原始位置
        local_path = self.base_dir / crop / readday / f'D{indicator}.csv'
        original_path = Path(f'F:/WaterML/{crop}/{readday}/D{indicator}.csv')

        if local_path.exists():
            return local_path
        elif original_path.exists():
            return original_path
        else:
            # 返回预期路径（会在错误信息中使用）
            return local_path

    def get_global_data_path(self, crop: str, readday: str) -> Path:
        """
        获取全球网格数据路径
        """
        local_path = self.base_dir / crop / readday / f'x_data_point_with_TotalN.{crop}.csv'
        original_path = Path(f'F:/WaterML/{crop}/{readday}/x_data_point_with_TotalN.{crop}.csv')

        if local_path.exists():
            return local_path
        elif original_path.exists():
            return original_path
        else:
            return local_path

    def get_reference_tif_path(self, crop: str, outday: str) -> Path:
        """
        获取参考TIF文件路径
        """
        local_path = self.base_dir / crop / outday / 'x_data_mask.tif'
        original_path = Path(f'F:/WaterML/{crop}/{outday}/x_data_mask.tif')

        if local_path.exists():
            return local_path
        elif original_path.exists():
            return original_path
        else:
            return local_path

    # =========================================================================
    # 输出数据路径
    # =========================================================================

    def get_response_surface_dir(self, crop: str, outday: str) -> Path:
        """获取响应面输出目录"""
        path = self.base_dir / crop / outday / 'response_surface'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_optimal_dir(self, crop: str, outday: str) -> Path:
        """获取最优结果输出目录"""
        path = self.base_dir / crop / outday / 'optimal'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_spatial_analysis_dir(self, crop: str, outday: str) -> Path:
        """获取空间分析输出目录"""
        path = self.base_dir / crop / outday / 'W' / 'spatial_analysis'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_mix_tnr_dir(self, crop: str, outday: str) -> Path:
        """获取中间结果目录（mix_TNR）"""
        path = self.base_dir / crop / outday / 'mix_TNR' / 'W'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_mix_ck_dir(self, crop: str, outday: str) -> Path:
        """获取对照组目录（mix_ck）"""
        path = self.base_dir / crop / outday / 'mix_ck' / 'W'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_mix_best_dir(self, crop: str, outday: str) -> Path:
        """获取最优结果目录（mix_best）"""
        path = self.base_dir / crop / outday / 'mix_best' / 'W'
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # 特定文件路径
    # =========================================================================

    def get_ck_file(self, crop: str, outday: str, indicator: str) -> Path:
        """获取对照组文件路径"""
        return self.get_mix_ck_dir(crop, outday) / f'{indicator}_ck.csv'

    def get_optimal_results_file(self, crop: str, outday: str) -> Path:
        """获取最优结果文件路径"""
        return self.get_optimal_dir(crop, outday) / 'optimal_ratio_results.csv'

    def get_surface_data_file(self, crop: str, outday: str) -> Path:
        """获取响应面数据文件路径"""
        return self.get_response_surface_dir(crop, outday) / 'surface_data.npy'

    # =========================================================================
    # 工具方法
    # =========================================================================

    def ensure_dir(self, path: Path) -> Path:
        """确保目录存在"""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_all_output_dirs(self, crop: str, outday: str) -> dict:
        """获取所有输出目录"""
        return {
            'response_surface': self.get_response_surface_dir(crop, outday),
            'optimal': self.get_optimal_dir(crop, outday),
            'spatial_analysis': self.get_spatial_analysis_dir(crop, outday),
            'mix_tnr': self.get_mix_tnr_dir(crop, outday),
            'mix_ck': self.get_mix_ck_dir(crop, outday),
            'mix_best': self.get_mix_best_dir(crop, outday),
        }


# =============================================================================
# 全局实例
# =============================================================================

# 默认路径管理器实例
path_manager = PathManager()


# =============================================================================
# 便捷函数
# =============================================================================

def get_path_manager(base_dir: Union[str, Path] = None) -> PathManager:
    """
    获取路径管理器实例

    参数：
        base_dir: 自定义基础目录

    返回：
        PathManager实例
    """
    return PathManager(base_dir) if base_dir else path_manager


def set_base_dir(base_dir: Union[str, Path]):
    """
    设置全局基础目录

    参数：
        base_dir: 新的基础目录路径
    """
    global BASE_DIR, path_manager
    BASE_DIR = Path(base_dir)
    path_manager = PathManager(BASE_DIR)
    print(f"基础输出目录已设置为: {BASE_DIR}")


# =============================================================================
# 测试函数
# =============================================================================

def test_path_manager():
    """测试路径管理器"""
    print("\n" + "="*70)
    print("  路径管理器测试")
    print("="*70)

    pm = get_path_manager()

    print(f"\n项目根目录: {PROJECT_ROOT}")
    print(f"基础输出目录: {pm.base_dir}")

    # 测试各种路径
    print(f"\n示例路径（crop=rice, outday=20251028）:")
    print(f"  响应面目录: {pm.get_response_surface_dir('rice', '20251028')}")
    print(f"  最优结果目录: {pm.get_optimal_dir('rice', '20251028')}")
    print(f"  空间分析目录: {pm.get_spatial_analysis_dir('rice', '20251028')}")

    print("\n" + "="*70)
    print("  测试通过！路径管理器工作正常")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_path_manager()