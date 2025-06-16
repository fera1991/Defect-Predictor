import pandas as pd
import re
import numpy as np
from metrics import extraer_metricas, map_repo_to_domain
from datetime import datetime

def clean_code(content):
    """Elimina comentarios y normaliza espacios en el código fuente."""
    try:
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
        return content
    except Exception as e:
        print(f"Error al limpiar código: {e}")
        return str(content)

def extraer_features(df_input):
    print("Extrayendo características de un batch de datos...")
    
    # Validar columnas necesarias
    required_cols = ['content_diff', 'content_full', 'datetime', 'methods', 'filepath', 'repo']
    if not all(col in df_input.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_input.columns]
        raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing_cols}")

    df = df_input.copy()
    
    # Calcular métricas para content_full y content_diff
    try:
        metrics_full = df['content_full'].apply(lambda x: extraer_metricas(x))
        metrics_diff = df['content_diff'].apply(lambda x: extraer_metricas(x))
    except Exception as e:
        print(f"Error al calcular métricas: {e}")
        return None

    # Convertir métricas a DataFrames
    metrics_full_df = pd.DataFrame(metrics_full.tolist()).add_suffix('_full')
    metrics_diff_df = pd.DataFrame(metrics_diff.tolist()).add_suffix('_diff')

    # Extraer características temporales
    try:
        df['commit_datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['commit_hour'] = df['commit_datetime'].dt.hour
        df['commit_day'] = df['commit_datetime'].dt.dayofweek
    except Exception as e:
        print(f"Error al extraer características temporales: {e}")
        df['commit_hour'] = 0
        df['commit_day'] = 0

    # Calcular longitud del filepath
    df['filepath_length'] = df['filepath'].str.len()

    # Calcular frecuencia de cambios por archivo
    try:
        file_change_freq = df.groupby(['repo', 'filepath'])['commit'].count().reset_index(name='file_change_freq')
        df = df.merge(file_change_freq, on=['repo', 'filepath'], how='left')
        df['file_change_freq'] = df['file_change_freq'].fillna(0)
    except Exception as e:
        print(f"Error al calcular file_change_freq: {e}")
        df['file_change_freq'] = 0

    # Mapear repositorio a dominio
    try:
        df['repo_domain'] = df['repo'].apply(map_repo_to_domain)
    except Exception as e:
        print(f"Error al mapear repo_domain: {e}")
        df['repo_domain'] = 'other'

    # Preparar textos para TF-IDF
    df['content_text_full'] = df['content_full'].apply(clean_code)
    df['content_text_diff'] = df['content_diff'].apply(clean_code)

    # Combinar todas las características
    features_df = pd.concat([
        metrics_full_df,
        metrics_diff_df,
        df[['commit_hour', 'commit_day', 'filepath_length', 'file_change_freq', 'repo_domain', 'content_text_full', 'content_text_diff']]
    ], axis=1)

    # Verificar columnas generadas
    expected_cols = (
        [col for col in metrics_full_df.columns] +
        [col for col in metrics_diff_df.columns] +
        ['commit_hour', 'commit_day', 'filepath_length', 'file_change_freq', 'repo_domain', 'content_text_full', 'content_text_diff']
    )
    missing_cols = [col for col in expected_cols if col not in features_df.columns]
    if missing_cols:
        print(f"Advertencia: Faltan columnas esperadas: {missing_cols}")
        for col in missing_cols:
            features_df[col] = 0  # Rellenar con 0 para columnas numéricas o vacías

    # Eliminar columna 'repo' explícitamente
    if 'repo' in features_df.columns:
        features_df = features_df.drop(columns=['repo'])
        print("Columna 'repo' eliminada correctamente.")
    else:
        print("Columna 'repo' no encontrada en features_df.")

    print("Columnas en features_df:", features_df.columns.tolist())
    print("Extracción de características completada.")
    return features_df