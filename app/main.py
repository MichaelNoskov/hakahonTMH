from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
import pandas as pd
import logging
from pathlib import Path

from app.data_processor import process_prediction_data
from app.model_loader import get_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wheel Wear Prediction API",
    description="API для предсказания износа колес локомотивов",
    version="1.0.0"
)


class WearDataItem(BaseModel):
    """Модель для одного элемента wear_data."""
    wheel_id: Optional[Union[int, str]] = None
    locomotive_series: str
    locomotive_number: Union[int, str]
    depo: str
    steel_num: Optional[float] = None
    mileage_start: Optional[float] = None

    @field_validator('locomotive_number')
    @classmethod
    def validate_locomotive_number(cls, v):
        """Преобразуем locomotive_number в строку для консистентности."""
        return str(v)


class ServiceDateItem(BaseModel):
    """Модель для одного элемента service_dates."""
    locomotive_series: str
    locomotive_number: Union[int, str]
    service_date: str
    service_type: Union[int, str]

    @field_validator('locomotive_number')
    @classmethod
    def validate_locomotive_number(cls, v):
        return str(v)


class PredictionRequest(BaseModel):
    """Модель запроса для предсказания."""
    wear_data: List[WearDataItem] = Field(..., min_length=1, description="Данные о колесах")
    service_dates: List[ServiceDateItem] = Field(default_factory=list, description="Данные о ремонтах")


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "Wheel Wear Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Предсказание износа колес",
            "/health": "GET - health check эндпоинт"
        }
    }


@app.get("/health")
async def health():
    """Health check эндпоинт."""
    try:
        model = get_model()
        model_status = "loaded"
    except Exception as e:
        logger.warning(f"Модель не загружена: {e}")
        model_status = "not_loaded"
    
    depo_mapping_path = Path("data/depo_mapping.csv")
    depo_mapping_status = "exists" if depo_mapping_path.exists() else "missing"
    
    status = "healthy" if model_status == "loaded" and depo_mapping_status == "exists" else "degraded"
    
    return {
        "status": status,
        "model": model_status,
        "depo_mapping": depo_mapping_status
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Эндпоинт для предсказания износа колес.
    
    Принимает:
    - wear_data: массив данных о колесах
    - service_dates: массив данных о ремонтах (опционально)
    
    Возвращает:
    - predictions: массив предсказаний wear_intensity для каждого колеса
    """
    try:
        logger.info(f"Получен запрос на предсказание для {len(request.wear_data)} колес")
        
        wear_data_dict = [item.model_dump() for item in request.wear_data]
        service_dates_dict = [item.model_dump() for item in request.service_dates]
        
        logger.info("Обработка данных...")
        df_features = process_prediction_data(
            wear_data=wear_data_dict,
            service_dates=service_dates_dict
        )
        
        logger.info(f"Подготовлено {len(df_features)} записей для предсказания")
        
        model = get_model()
        
        logger.info("Выполнение предсказаний...")
        predictions = model.predict(df_features)
        
        predictions_list = predictions.tolist()
        
        logger.info(f"Предсказания выполнены для {len(predictions_list)} колес")
        
        return JSONResponse(content=predictions_list)
        
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка конфигурации: {str(e)}")
    
    except ValueError as e:
        logger.error(f"Ошибка валидации данных: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка валидации данных: {str(e)}")
    
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
