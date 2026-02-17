"""
Модуль загрузки и кэширования CatBoost модели.
"""
from pathlib import Path
from catboost import CatBoostRegressor
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_model: Optional[CatBoostRegressor] = None


def load_model(model_path: str = "model/model.cbm") -> CatBoostRegressor:
    """
    Загружает CatBoost модель из файла с кэшированием.
    При первом вызове загружает модель, при последующих возвращает кэшированную.
    
    Args:
        model_path: Путь к файлу модели (.cbm)
        
    Returns:
        Загруженная CatBoost модель
        
    Raises:
        FileNotFoundError: Если файл модели не найден
        Exception: При ошибке загрузки модели
    """
    global _model
    
    if _model is not None:
        logger.info("Используется кэшированная модель")
        return _model
    
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {model_path}. "
            f"Убедитесь, что модель сохранена из baseline.ipynb"
        )
    
    logger.info(f"Загрузка модели из {model_path}")
    
    try:
        _model = CatBoostRegressor()
        _model.load_model(str(model_file))
        logger.info("Модель успешно загружена")
        return _model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise


def get_model() -> CatBoostRegressor:
    """
    Получает загруженную модель (загружает при первом вызове).
    
    Returns:
        CatBoost модель
    """
    return load_model()


def clear_cache():
    """
    Очищает кэш модели (для тестирования или перезагрузки).
    """
    global _model
    _model = None
    logger.info("Кэш модели очищен")
