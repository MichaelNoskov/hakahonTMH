import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def load_depo_mapping(data_path: str = "data/depo_mapping.csv") -> pd.DataFrame:
    depo_path = Path(data_path)
    if not depo_path.exists():
        raise FileNotFoundError(f"Файл depo_mapping.csv не найден: {depo_path}")
    
    df_depos = pd.read_csv(depo_path, encoding='utf-8')
    return df_depos[['depo', 'directorate']]


def process_service_dates(
    service_dates: List[Dict[str, Any]],
    end_date: str = '2025-09-30'
) -> pd.DataFrame:
    if not service_dates:
        return pd.DataFrame(columns=['locomotive_number', 'total_service_type_4'])
    
    df_service = pd.DataFrame(service_dates)
    
    df_service['service_type'] = np.where(
        df_service['service_type'] == 'обточка',
        4,
        df_service['service_type']
    )
    df_service['service_type'] = df_service['service_type'].astype(str)
    
    df_service['service_date'] = pd.to_datetime(df_service['service_date'])
    df_service = df_service.sort_values(['locomotive_number', 'service_date']).reset_index(drop=True)
    
    current_date = pd.Timestamp(end_date)
    
    def create_service_features(group):
        """Создает признаки для группы service_dates по locomotive_number."""
        dates = group['service_date'].values
        types = group['service_type'].values.astype(int)
        
        result = {
            'locomotive_number': group.name,
        }
        
        for t in range(1, 5):
            count = sum(1 for typ in types if typ == t)
            result[f'total_service_type_{t}'] = count
        
        return pd.Series(result)
    
    df_service_features = df_service.groupby('locomotive_number').apply(
        create_service_features
    ).reset_index(drop=True)
    
    return df_service_features


def process_wear_data(
    wear_data: List[Dict[str, Any]],
    depo_mapping: pd.DataFrame
) -> pd.DataFrame:
    if not wear_data:
        raise ValueError("wear_data не может быть пустым")
    
    df_wheels = pd.DataFrame(wear_data)
    
    df_wheels = df_wheels.merge(
        depo_mapping[['depo', 'directorate']],
        on='depo',
        how='left'
    )
    
    df_wheels['directorate'] = df_wheels['directorate'].fillna('missing')
    
    return df_wheels


def merge_and_prepare_features(
    df_wheels: pd.DataFrame,
        df_service_features: pd.DataFrame
) -> pd.DataFrame:
    df = df_wheels.merge(
        df_service_features,
        on='locomotive_number',
        how='left'
    )
    
    service_columns = [col for col in df_service_features.columns 
                      if col != 'locomotive_number']
    
    for col in service_columns:
        if col in df.columns:
            if 'total_' in col:
                df[col] = df[col].fillna(0)
    
    required_features = [
        'locomotive_series',
        'locomotive_number',
        'depo',
        'directorate',
        'steel_num',
        'mileage_start',
        'total_service_type_4'
    ]
    
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные признаки: {missing}")
    
    df['locomotive_number'] = df['locomotive_number'].astype(str)
    
    df_features = df[required_features].copy()
    
    return df_features


def process_prediction_data(
    wear_data: List[Dict[str, Any]],
    service_dates: List[Dict[str, Any]],
    data_path: str = "data/depo_mapping.csv",
    end_date: str = '2025-09-30'
) -> pd.DataFrame:
    depo_mapping = load_depo_mapping(data_path)
    
    df_service_features = process_service_dates(service_dates, end_date)
    
    df_wheels = process_wear_data(wear_data, depo_mapping)
    
    df_features = merge_and_prepare_features(df_wheels, df_service_features)
    
    return df_features
