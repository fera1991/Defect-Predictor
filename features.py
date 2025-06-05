import pandas as pd
import re
from metrics import extraer_metricas, map_repo_to_domain

def clean_code(content):
    """Elimina comentarios y normaliza espacios en el código fuente."""
    # Eliminar comentarios de una línea
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
    # Normalizar espacios y eliminar líneas vacías
    content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
    return content

def extraer_features(df_input):
    print("Extrayendo características de un batch de datos...")
    
    # Validar columnas necesarias
    required_cols = ['content_diff', 'content_full', 'datetime', 'methods', 'filepath', 'repo']
    if not all(col in df_input.columns for col in required_cols):
        raise ValueError("Faltan columnas requeridas en el DataFrame.")
    
    df = df_input.copy()
    
    try:
        # Convertir contenido a texto
        df['content_diff'] = df['content_diff'].apply(
            lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x))
        df['content_full'] = df['content_full'].apply(
            lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x))
        
        # Limpiar el código
        df['content_diff'] = df['content_diff'].apply(clean_code)
        df['content_full'] = df['content_full'].apply(clean_code)
        
        # Extraer características temporales
        df['commit_datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['commit_hour'] = df['commit_datetime'].dt.hour
        df['commit_day'] = df['commit_datetime'].dt.dayofweek
        
        # Imputación para commit_hour y commit_day
        median_hour = df['commit_hour'].median()
        median_day = df['commit_day'].median()
        df['commit_hour'] = df['commit_hour'].fillna(median_hour)
        df['commit_day'] = df['commit_day'].fillna(median_day)
        
        # Extraer número de métodos
        df['num_methods'] = df['methods'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Longitud del filepath
        df['filepath_length'] = df['filepath'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        # Frecuencia de cambio del archivo
        df['file_change_freq'] = df.groupby('filepath')['filepath'].transform('count')
        
        # Extraer métricas de contenido
        metricas_full = df['content_full'].apply(extraer_metricas).apply(pd.Series).add_suffix('_full')
        metricas_diff = df['content_diff'].apply(extraer_metricas).apply(pd.Series).add_suffix('_diff')
        
        # Concatenar todas las características
        df_features = pd.concat([
            metricas_full,
            metricas_diff,
            df[['commit_hour', 'commit_day', 'num_methods', 'filepath_length', 'repo', 'file_change_freq']]
        ], axis=1)
        
        # Agregar texto del contenido y dominio del repositorio
        df_features['content_text_full'] = df['content_full']
        df_features['content_text_diff'] = df['content_diff']
        df_features['repo_domain'] = df['repo'].apply(map_repo_to_domain)
        
        print("Extracción de características completada.")
        return df_features
    
    except Exception as e:
        print(f"Error en extraer_features: {e}")
        return None