import pandas as pd
import time
import math
from collections import Counter
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Scikit-learn y otras bibliotecas
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef

#pylint
import tempfile
import subprocess
import json

# ==============================
# Configuración de Parámetros
# ==============================

# Parámetros Generales
GENERAL_PARAMS = {
    'RANDOM_STATE': 42,
    'DATA_FRACTION': 1.0,  # Recomendación: Aumentar a 0.5 o 1.0 tras validar
    'DATA_TYPE': 1,
}

# Parámetros para TfidfVectorizer
TFIDF_PARAMS = {
    'max_features': 20000,
    'ngram_range': (1, 3),
    'min_df': 5,  # Ajustado para reducir ruido
    'max_df': 0.6,
    'stop_words': ['def', 'import', 'class', 'if', 'for', 'while', 'return', 'try', 'except'],  # Ampliado
    'lowercase': True,
    'analyzer': 'word',
    'norm': 'l2',
    'sublinear_tf': True  # Activado para mejorar TF-IDF
}

# Parámetros para TruncatedSVD
SVD_PARAMS = {
    'n_components': 150 ,  # Aumentado para capturar más varianza
    'algorithm': 'randomized',
    'n_iter': 10,
    'random_state': GENERAL_PARAMS['RANDOM_STATE']
}

# Parámetros para RandomForestClassifier
RF_PARAMS = {
    'n_estimators': 500,  # Reducido para acelerar entrenamiento
    'max_depth': 30,  # Limitado para evitar sobreajuste
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',  # Cambiado para mejor generalización
    'class_weight': 'balanced',
    'random_state': GENERAL_PARAMS['RANDOM_STATE'],
    'n_jobs': 4
}

# Parámetros para SimpleImputer
IMPUTER_PARAMS = {
    'strategy': 'median'
}

# Parámetros para StandardScaler
SCALER_PARAMS = {
    'with_mean': True,
    'with_std': True
}

# Parámetros para OneHotEncoder
OHE_PARAMS = {
    'handle_unknown': 'ignore'
}

# ==============================
# Funciones de extracción de métricas
# ==============================

def token_entropy(content):
    tokens = content.split()
    if not tokens:
        return 0
    freqs = Counter(tokens)
    total = len(tokens)
    return -sum((count / total) * math.log2(count / total) for count in freqs.values())

def num_indent_levels(content):
    return len(set(len(line) - len(line.lstrip(' ')) for line in content.splitlines() if line.strip()))

def has_mixed_tabs_spaces(content):
    lines = content.splitlines()
    return int(any('\t' in line and line.startswith(' ') for line in lines))

def has_unclosed_parens(content):
    return int(content.count('(') != content.count(')'))

def repetition_score(content):
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return 0
    repeated = len(lines) - len(set(lines))
    return repeated / len(lines)

def num_decision_points(content):
    keywords = ['if ', 'for ', 'while ', 'try:', 'except', 'elif', 'and ', 'or ']
    return sum(content.count(kw) for kw in keywords)

def num_external_calls(content):
    lines = content.splitlines()
    loc = len(lines) or 1
    # Normalizar llamadas externas por número de líneas
    return content.count('.') / loc

def map_repo_to_domain(repo):
    mapping = {
        "lightning": "deep_learning", "ansible": "automation", "airflow": "automation",
        "celery": "task_queue", "openpilot": "autonomous_driving", "django": "web_framework",
        "django-rest-framework": "web_framework", "freqtrade": "trading", "jax": "deep_learning",
        "core": "iot", "transformers": "deep_learning", "numpy": "data_science",
        "pandas": "data_science", "pipenv": "development", "poetry": "development",
        "ray": "machine_learning", "redash": "data_science", "scikit-learn": "machine_learning",
        "scrapy": "web_scraping", "sentry": "logging", "spacy": "nlp", "yolov5": "image_processing",
        "black": "development"
    }
    repo_lower = repo.lower()
    for key, value in mapping.items():
        if key in repo_lower:
            return value
    return "other"

def map_path_to_purpose(filepath):
    path = filepath.lower()
    if any(p in path for p in ["/test", "/tests", "/__tests__", "/unittest"]):
        return "testing"
    if any(p in path for p in ["/doc", "/docs", "/documentation"]):
        return "documentation"
    if any(p in path for p in ["/example", "/examples", "/demo"]):
        return "examples"
    if any(p in path for p in ["/util", "/utils", "/helpers"]):
        return "utilities"
    if any(p in path for p in ["/config", "/configs", ".github", ".circleci", "/ci"]):
        return "configuration"
    if "/src" in path:
        return "source"
    if path.count("/") <= 2:
        return "root_or_meta"
    return "other"

def avg_indent_length(content):
    lines = content.splitlines()
    indents = [len(line) - len(line.lstrip(' ')) for line in lines if line.strip()]
    return sum(indents) / len(indents) if indents else 0

def max_indent_length(content):
    lines = content.splitlines()
    indents = [len(line) - len(line.lstrip(' ')) for line in lines if line.strip()]
    return max(indents) if indents else 0

def num_todo_comments(content):
    lines = content.splitlines()
    return sum(1 for line in lines if '#' in line and ('todo' in line.lower() or 'fixme' in line.lower()))

def count_operators(content):
    operators = ['+', '-', '*', '/', '%', '**', '//']
    return sum(content.count(op) for op in operators)

def avg_words_per_line(content):
    lines = content.splitlines()
    words_per_line = [len(line.split()) for line in lines if line.strip()]
    return sum(words_per_line) / len(words_per_line) if words_per_line else 0

def num_docstrings(content):
    return (content.count('"""') // 2) + (content.count("'''") // 2)

def num_error_keywords(content):
    error_keywords = ['error', 'fail', 'exception', 'bug']
    content_lower = content.lower()
    loc = len(content_lower.splitlines()) or 1
    # Normalizar ocurrencias de palabras clave de error
    return sum(content_lower.count(word) for word in error_keywords) / loc

def extraer_metricas(content):
    try:    
        lines = content.splitlines()
        loc = len(lines)
        num_comments = content.count('#')
        num_empty_lines = sum(1 for l in lines if l.strip() == '')
        avg_line_length = sum(len(line) for line in lines) / loc if loc > 0 else 0
        comment_ratio = num_comments / loc if loc > 0 else 0
        empty_ratio = num_empty_lines / loc if loc > 0 else 0
        code_density = (loc - num_empty_lines) / loc if loc > 0 else 0
        num_todos = num_todo_comments(content)
        todo_ratio = num_todos / loc if loc > 0 else 0

        # === Análisis estático con pylint ===
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ['pylint', tmp_path, '-f', 'json', '--disable=all', '--enable=E,W,C,R'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            pylint_output = json.loads(result.stdout) if result.stdout else []
        except Exception:
            pylint_output = []

        conteo_violaciones = {
            "error": 0,
            "warning": 0,
            "convention": 0,
            "refactor": 0
        }
        for v in pylint_output:
            t = v.get("type")
            if t in conteo_violaciones:
                conteo_violaciones[t] += 1

        return {
            "loc": loc,
            "num_defs": content.count('def '),
            "num_classes": content.count('class '),
            "num_imports": content.count('import '),
            "has_try_except": int('try' in content and 'except' in content),
            "has_if_else": int('if ' in content and 'else' in content),
            "uses_torch": int('torch' in content),
            "uses_numpy": int('numpy' in content),
            "uses_cv2": int('cv2' in content),
            "has_return": int('return' in content),
            "has_raise": int('raise' in content),
            "num_comments": num_comments,
            "num_empty_lines": num_empty_lines,
            "length": len(content),
            "avg_line_length": avg_line_length,
            "comment_ratio": comment_ratio,
            "empty_ratio": empty_ratio,
            "code_density": code_density,
            "token_entropy": token_entropy(content),
            "num_indent_levels": num_indent_levels(content),
            "has_tab_mix": has_mixed_tabs_spaces(content),
            "has_unclosed_parens": has_unclosed_parens(content),
            "repetition_score": repetition_score(content),
            "num_decision_points": num_decision_points(content),
            "num_external_calls": num_external_calls(content),
            "avg_indent_length": avg_indent_length(content),
            "max_indent_length": max_indent_length(content),
            "num_todo_comments": num_todos,
            "num_operators": count_operators(content),
            "avg_words_per_line": avg_words_per_line(content),
            "num_docstrings": num_docstrings(content),
            "num_error_keywords": num_error_keywords(content),
            "todo_ratio": todo_ratio,
            "num_pylint_errors": conteo_violaciones["error"],
            "num_pylint_warnings": conteo_violaciones["warning"],
            "num_pylint_conventions": conteo_violaciones["convention"],
            "num_pylint_refactors": conteo_violaciones["refactor"]
        }
    except Exception as e:
        return {
            "loc": 0,
            "num_defs": 0,
            "num_classes": 0,
            "num_imports": 0,
            "has_try_except": 0,
            "has_if_else": 0,
            "uses_torch": 0,
            "uses_numpy": 0,
            "uses_cv2": 0,
            "has_return": 0,
            "has_raise": 0,
            "num_comments": 0,
            "num_empty_lines": 0,
            "length": 0,
            "avg_line_length": 0,
            "comment_ratio": 0,
            "empty_ratio": 0,
            "code_density": 0,
            "token_entropy": 0,
            "num_indent_levels": 0,
            "has_tab_mix": 0,
            "has_unclosed_parens": 0,
            "repetition_score": 0,
            "num_decision_points": 0,
            "num_external_calls": 0,
            "avg_indent_length": 0,
            "max_indent_length": 0,
            "num_todo_comments": 0,
            "num_operators": 0,
            "avg_words_per_line": 0,
            "num_docstrings": 0,
            "num_error_keywords": 0,
            "todo_ratio": 0,
            "num_pylint_errors": 0,
            "num_pylint_warnings": 0,
            "num_pylint_conventions": 0,
            "num_pylint_refactors": 0
        }

def extraer_features(df_input):
    print("Extrayendo características de un batch de datos...")
    try:
        df_input['content'] = df_input['content'].apply(
            lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x))
        df_input['commit_datetime'] = pd.to_datetime(df_input['datetime'], errors='coerce')
        df_input['commit_hour'] = df_input['commit_datetime'].dt.hour
        df_input['commit_day'] = df_input['commit_datetime'].dt.dayofweek
        # —————————— IMPUTACIÓN PARA commit_hour Y commit_day ——————————
        median_hour = df_input['commit_hour'].median()
        median_day  = df_input['commit_day'].median()
        df_input['commit_hour'] = df_input['commit_hour'].fillna(median_hour)
        df_input['commit_day'] = df_input['commit_day'].fillna(median_day)
        df_input['num_methods'] = df_input['methods'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df_input['filepath_length'] = df_input['filepath'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        #df_input['file_purpose'] = df_input['filepath'].apply(map_path_to_purpose)
        df_input['file_change_freq'] = df_input['filepath'].map(df_input['filepath'].value_counts())
        
        metricas = df_input['content'].apply(extraer_metricas).apply(pd.Series)
        df_features = pd.concat([
            metricas, 
            df_input[['commit_hour', 'commit_day', 'num_methods', 'filepath_length', 'repo', 'file_change_freq']]
        ], axis=1)
        
        df_features['content_text'] = df_input['content']
        df_features['repo_domain'] = df_input['repo'].apply(map_repo_to_domain)
        print("Extracción de características completada.")
        return df_features
    except Exception as e:
        print(f"Error en extraer_features: {e}")
        return None

# ==============================
# Código principal
# ==============================

def encontrar_umbral_optimo_mcc(y_true, y_prob, n_steps=100):
    best_mcc = -1
    best_thresh = 0.5
    thresholds = np.linspace(0, 1, n_steps)
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_t)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
    return best_thresh, best_mcc


if __name__ == "__main__":
    start = time.time()
    
    # Cargar los datasets
    print("Cargando los datasets...")
    df_train = pd.read_parquet("jit_bug_prediction_splits/random/train.parquet.gzip")
    df_test = pd.read_parquet("jit_bug_prediction_splits/random/test.parquet.gzip")
    df_val = pd.read_parquet("jit_bug_prediction_splits/random/val.parquet.gzip")
    print("Datasets cargados.")

    # Procesar los datos según el valor de data_type
    if GENERAL_PARAMS['DATA_TYPE'] == 0:
        # Tipo 0: Usar datasets originales sin modificar ni estratificar
        print("\nTipo 0: Usar datasets originales sin modificar ni estratificar")
        df_train['label'] = df_train['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        df_test['label'] = df_test['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        df_val['label'] = df_val['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
    elif GENERAL_PARAMS['DATA_TYPE'] == 1:
        # Tipo 1: Combinar, estratificar y muestrear
        print("\nTipo 1: Combinar, estratificar y muestrear")
        # Combinar datasets
        data = pd.concat([df_train, df_test, df_val]).reset_index(drop=True)
        # Crear columna 'label'
        data['label'] = data['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        # Mostrar distribución de clases original
        print("\nDistribución de clases original (antes de muestreo):")
        print(data['label'].value_counts(normalize=True).to_string())
        # Estratificar división
        print("\nPreparando conjuntos de datos con estratificación...")
        X = data.drop(columns=['label'])
        y = data['label']
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        # Aplicar muestreo a los conjuntos estratificados
        X_train = X_train.sample(frac=GENERAL_PARAMS['DATA_FRACTION'], random_state=GENERAL_PARAMS['RANDOM_STATE'])
        y_train = y_train.loc[X_train.index]
        X_test = X_test.sample(frac=GENERAL_PARAMS['DATA_FRACTION'], random_state=GENERAL_PARAMS['RANDOM_STATE'])
        y_test = y_test.loc[X_test.index]
        X_val = X_val.sample(frac=GENERAL_PARAMS['DATA_FRACTION'], random_state=GENERAL_PARAMS['RANDOM_STATE'])
        y_val = y_val.loc[X_val.index]
        # Convertir a DataFrames
        df_train = X_train.copy()
        df_train['label'] = y_train
        df_test = X_test.copy()
        df_test['label'] = y_test
        df_val = X_val.copy()
        df_val['label'] = y_val
    else:
        raise ValueError("data_type debe ser 0 o 1")

    # Convertir la columna 'content' en todos los casos
    df_train['content'] = df_train['content'].apply(
        lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x)
    )
    df_test['content'] = df_test['content'].apply(
        lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x)
    )
    df_val['content'] = df_val['content'].apply(
        lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x)
    )

    # Definir X y y para cada conjunto de manera consistente
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']
    X_val = df_val.drop(columns=['label'])
    y_val = df_val['label']

    # Mostrar tamaños de los conjuntos
    print(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
    print(f"Tamaño del conjunto de prueba: {len(df_test)}")
    print(f"Tamaño del conjunto de validación: {len(df_val)}")
    print("Conjuntos de datos preparados.")
        
    # Definir numeric_features y categorical_features antes de la extracción
    numeric_features = [
        "loc", "num_defs", "num_classes", "num_imports", "has_try_except", "has_if_else",
        "uses_torch", "uses_numpy", "uses_cv2", "has_return", "has_raise", "num_comments",
        "num_empty_lines", "length", "avg_line_length", "comment_ratio", "empty_ratio",
        "code_density", "token_entropy", "num_indent_levels", "has_tab_mix", "has_unclosed_parens",
        "repetition_score", "num_decision_points", "num_external_calls",
        "commit_hour", "commit_day", "num_methods", "filepath_length", "file_change_freq",
        "avg_indent_length", "max_indent_length", "num_todo_comments", "num_operators",
        "avg_words_per_line", "num_docstrings", "num_error_keywords", "todo_ratio",
        # 🔽 Nuevas métricas de pylint
        "num_pylint_errors", "num_pylint_warnings", "num_pylint_conventions", "num_pylint_refactors"
    ]

    categorical_features = ['repo_domain']  # Lista inicial de características categóricas

    print("\nExtrayendo características para el conjunto de entrenamiento...")
    start_extract = time.time()
    X_train_feats = extraer_features(X_train)
    if X_train_feats is None:
        print("Error: No se pudieron extraer características para el conjunto de entrenamiento.")
        exit(1)
    end_extract = time.time()
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")

    print("\nExtrayendo características para el conjunto de prueba...")
    start_extract = time.time()
    X_test_feats = extraer_features(X_test)
    if X_test_feats is None:
        print("Error: No se pudieron extraer características para el conjunto de prueba.")
        exit(1)
    end_extract = time.time()
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")

    print("\nExtrayendo características para el conjunto de validación...")
    start_extract = time.time()
    X_val_feats = extraer_features(X_val)
    if X_val_feats is None:
        print("Error: No se pudieron extraer características para el conjunto de validación.")
        exit(1)
    end_extract = time.time()
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")

    # Mostrar información de características
    print(f"\nNúmero de características extraídas: {len(X_train_feats.columns)}")
    print(f"Valores faltantes por característica:\n{X_train_feats.isna().sum()}")

    # Depuración: Mostrar distribución de 'file_purpose' antes de eliminar constantes
    #print("\nDistribución de 'file_purpose' en el conjunto de entrenamiento:")
    #print(X_train_feats['file_purpose'].value_counts().to_string())

    # Identificar y eliminar características constantes
    constant_features = X_train_feats.columns[X_train_feats.nunique() == 1]
    print("\nCaracterísticas constantes:", constant_features)
    if len(constant_features) > 0:
        X_train_feats = X_train_feats.drop(columns=constant_features)
        X_test_feats = X_test_feats.drop(columns=constant_features)
        X_val_feats = X_val_feats.drop(columns=constant_features)
        numeric_features = [f for f in numeric_features if f not in constant_features]
        categorical_features = [f for f in categorical_features if f not in constant_features]
        print(f"Características constantes eliminadas: {len(constant_features)}")

    # Depuración: Mostrar columnas disponibles y categorical_features
    print("\nColumnas en X_train_feats después de eliminar constantes:", X_train_feats.columns.tolist())
    print("Características categóricas actualizadas:", categorical_features)
    

    print("Definiendo el preprocesador de datos...")

    # Crear transformadores para el ColumnTransformer
    transformers = [
        ('num', SkPipeline(steps=[
            # Primero imputar commit_hour/day con la mediana
            ('imputer', SimpleImputer(**IMPUTER_PARAMS)),
            ('scaler', StandardScaler(**SCALER_PARAMS)),
            ('variance', VarianceThreshold(threshold=0)),  # Elimina constantes
            ('select', SelectKBest(f_classif, k=20))
        ]), numeric_features),
        ('text', SkPipeline(steps=[
            ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
            ('svd', TruncatedSVD(**SVD_PARAMS))
        ]), 'content_text')
    ]

    # Añadir transformador categórico solo si hay columnas categóricas
    if categorical_features:
        transformers.append(
            ('cat', OneHotEncoder(**OHE_PARAMS), categorical_features)
        )
    else:
        print("Advertencia: No hay características categóricas disponibles. Se omite el transformador categórico.")

    preprocessor = ColumnTransformer(transformers=transformers)

    
    # Definición de los clasificadores base
    clf_rf = RandomForestClassifier(**RF_PARAMS)
    clf_xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=GENERAL_PARAMS['RANDOM_STATE']
    )
    clf_lr = LogisticRegression(max_iter=1000, random_state=GENERAL_PARAMS['RANDOM_STATE'])

    # VotingClassifier (votación suave con probabilidades)
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', clf_rf),
            ('xgb', clf_xgb),
            ('lr', clf_lr)
        ],
        voting='soft'
    )

    print("Pipeline con VotingClassifier")
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=GENERAL_PARAMS['RANDOM_STATE'])),
        ('clf', CalibratedClassifierCV(voting_clf, cv=3, method='sigmoid'))
    ])

    #print("Construyendo el pipeline completo con Random Forest...")
    #pipeline = ImbPipeline(steps=[
    #    ('preprocessor', preprocessor),
    #    ('smote', SMOTE(random_state=GENERAL_PARAMS['RANDOM_STATE'])),
    #    ('clf', CalibratedClassifierCV(RandomForestClassifier(**RF_PARAMS), cv=3, method='sigmoid'))
    #])

    #pipeline = SkPipeline(steps=[
    #    ('preprocessor', preprocessor),
    #    ('clf', RandomForestClassifier(**RF_PARAMS))
    #])
    
    print("Entrenando el modelo...")

    param_dist = {
        # RandomForestClassifier
        'clf__estimator__rf__n_estimators': [100, 200, 500],
        'clf__estimator__rf__max_depth': [None, 10, 20, 30],
        'clf__estimator__rf__min_samples_leaf': [1, 2, 4],
        'clf__estimator__rf__min_samples_split': [2, 5, 10],
        'clf__estimator__rf__max_features': ['sqrt', 'log2', None],
        'clf__estimator__rf__class_weight': [None, 'balanced', 'balanced_subsample'],

        # XGBClassifier
        'clf__estimator__xgb__n_estimators': [100, 200, 300],
        'clf__estimator__xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__estimator__xgb__max_depth': [3, 5, 7, 10],
        'clf__estimator__xgb__subsample': [0.6, 0.8, 1.0],
        'clf__estimator__xgb__colsample_bytree': [0.6, 0.8, 1.0],

        # LogisticRegression
        'clf__estimator__lr__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'clf__estimator__lr__penalty': ['l2'],
        # 'clf__estimator__lr__solver': ['lbfgs'],  # opcional si agregas más penalties

        # TF-IDF + SVD (solo si incluyes estos en el pipeline)
        'preprocessor__text__tfidf__max_df': [0.7, 0.85, 1.0],
        'preprocessor__text__tfidf__min_df': [1, 3, 5],
        'preprocessor__text__svd__n_components': [20, 50, 100]
    }


    # Configuración de RandomizedSearchCV
    #random_search = RandomizedSearchCV(
    #    estimator=pipeline,
    #    param_distributions=param_dist,
    #    n_iter=5,                    # 5 combinaciones aleatorias
    #    cv=3,                        # 3 folds
    #    scoring='f1_macro',
    #    n_jobs=4,
    #    random_state=GENERAL_PARAMS['RANDOM_STATE'],
    #    verbose=1
    #)

    #print("Buscando hiperparámetros con RandomizedSearchCV...")
    #start_train = time.time()
    #random_search.fit(X_train_feats, y_train)
    #end_train = time.time()
    #print(f"🕒 Tiempo de RandomizedSearchCV: {end_train - start_train:.2f} segundos")

    # Mejor pipeline encontrado
    #best_pipeline = random_search.best_estimator_
    #print("Mejores parámetros encontrados:", random_search.best_params_)

    print("Entrenando el modelo con hiperparámetros fijos...")
    pipeline.set_params(
        preprocessor__text__tfidf__min_df=3,
        preprocessor__text__tfidf__max_df=1.0,
        preprocessor__text__svd__n_components=100,

        clf__estimator__xgb__subsample=1.0,
        clf__estimator__xgb__n_estimators=300,
        clf__estimator__xgb__max_depth=10,
        clf__estimator__xgb__learning_rate=0.05,
        clf__estimator__xgb__colsample_bytree=0.6,

        clf__estimator__rf__n_estimators=100,
        clf__estimator__rf__min_samples_split=10,
        clf__estimator__rf__min_samples_leaf=4,
        clf__estimator__rf__max_features='sqrt',
        clf__estimator__rf__max_depth=None,
        clf__estimator__rf__class_weight='balanced',

        clf__estimator__lr__penalty='l2',
        clf__estimator__lr__C=1.0
    )

    start_train = time.time()
    pipeline.fit(X_train_feats, y_train)
    end_train = time.time()
    print(f"\U0001F552 Tiempo de entrenamiento: {end_train - start_train:.2f} segundos")

    # Usar directamente pipeline como best_pipeline
    best_pipeline = pipeline

    #start_train = time.time()
    #pipeline.fit(X_train_feats, y_train)
    #end_train = time.time()
    #print(f"🕒 Tiempo de entrenamiento: {end_train - start_train:.2f} segundos")
    
    print("Realizando predicciones en el conjunto de prueba...")
    start_pred = time.time()
    y_pred = best_pipeline.predict(X_test_feats)
    y_prob = best_pipeline.predict_proba(X_test_feats)[:, 1]
    end_pred = time.time()
    print(f"🕒 Tiempo de predicción: {end_pred - start_pred:.2f} segundos")
    
    # --- Distribución de Clases ---
    print("\n=== Distribución de Clases ===")
    print("Entrenamiento:\n", y_train.value_counts(normalize=True).to_string())
    print("\nPrueba:\n", y_test.value_counts(normalize=True).to_string())
    print("\nValidación:\n", y_val.value_counts(normalize=True).to_string())

    # --- Validación Cruzada ---
    #print("\n=== Validación Cruzada (Entrenamiento) ===")
    #start_pred = time.time()
    #cv_scores = cross_val_score(best_pipeline, X_train_feats, y_train, cv=5, scoring='f1_macro')
    #print(f"Puntuaciones F1-macro (5-fold CV): {cv_scores}")
    #print(f"Media: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
    #end_pred = time.time()
    #print(f"🕒 Tiempo de Validación Cruzada: {end_pred - start_pred:.2f} segundos")

    # --- Evaluación en Conjunto de Prueba ---
    print("\n=== Evaluación en Conjunto de Prueba ===")
    print("Realizando predicciones en el conjunto de prueba...")
    start_pred = time.time()
    y_pred = best_pipeline.predict(X_test_feats)
    y_prob = best_pipeline.predict_proba(X_test_feats)[:, 1]
    end_pred = time.time()
    print(f"🕒 Tiempo de predicción: {end_pred - start_pred:.2f} segundos")

    umbral_mcc, mcc_val = encontrar_umbral_optimo_mcc(y_test, y_prob)
    print(f"🔍 Umbral óptimo (MCC): {umbral_mcc:.3f}, MCC: {mcc_val:.3f}")

    # Aplico el umbral óptimo
    y_pred_mcc = (y_prob >= umbral_mcc).astype(int)

    # Reporte con MCC
    print("\n📊 Reporte de Clasificación (umbral óptimo MCC):")
    print(classification_report(y_test, y_pred_mcc))
    print("Matriz de Confusión (umbral óptimo MCC):")
    print(confusion_matrix(y_test, y_pred_mcc))

    # Reporte de Clasificación (umbral=0.5)
    #print("\n📊 Reporte de Clasificación (umbral=0.5):")
    #print(classification_report(y_test, y_pred))

    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión (umbral=0.5):")
    print(cm)

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Área bajo la curva ROC: {roc_auc:.4f}")

    # Selección automática de umbral óptimo
    print("\n🔍 Selección de Umbral Óptimo (máximo F1-score para clase '1'):")
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_pr[optimal_idx]
    print(f"Umbral óptimo: {optimal_threshold:.4f}")
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    print(f"Reporte de Clasificación (umbral={optimal_threshold:.4f}):")
    print(classification_report(y_test, y_pred_optimal))
    print("Matriz de Confusión (umbral óptimo):")
    print(confusion_matrix(y_test, y_pred_optimal))

    # Evaluación con Diferentes Umbrales
    print("\n🔍 Evaluación con Diferentes Umbrales para Clase 1:")
    thresholds = [0.3,0.4,0.5, 0.55, 0.6, 0.7, 0.8, 0.9]  # Añadido 0.5 para comparación
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        print(f"\nUmbral: {thresh}")
        print(classification_report(y_test, y_pred_thresh))
        print("Matriz de Confusión:")
        print(confusion_matrix(y_test, y_pred_thresh))

    # Distribución de Probabilidades Predichas
    print("\n📈 Distribución de Probabilidades Predichas (Clase 1):")
    prob_stats = pd.Series(y_prob).describe()
    print(prob_stats.to_string())
    plt.figure(figsize=(8, 6))
    plt.hist(y_prob, bins=20, density=True, color='skyblue', edgecolor='black')
    plt.title("Distribución de Probabilidades Predichas (Clase 1)")
    plt.xlabel("Probabilidad")
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.show()

    # --- Evaluación en Conjunto de Validación ---
    print("\n=== Evaluación en Conjunto de Validación ===")
    print("Realizando predicciones en el conjunto de validación...")
    start_pred = time.time()
    y_val_pred = best_pipeline.predict(X_val_feats)
    y_val_prob = best_pipeline.predict_proba(X_val_feats)[:, 1]
    end_pred = time.time()
    print(f"🕒 Tiempo de predicción: {end_pred - start_pred:.2f} segundos")

    # Reporte de Clasificación (umbral=0.5)
    print("\n📊 Reporte de Clasificación (umbral=0.5):")
    print(classification_report(y_val, y_val_pred))

    # Matriz de Confusión
    cm_val = confusion_matrix(y_val, y_val_pred)
    print("Matriz de Confusión (umbral=0.5):")
    print(cm_val)

    # Curva ROC
    fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_prob)
    roc_auc_val = roc_auc_score(y_val, y_val_prob)
    print(f"Área bajo la curva ROC: {roc_auc_val:.4f}")

    # --- Visualizaciones ---
    # Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Prueba (AUC = {roc_auc:.4f})', color='blue')
    plt.plot(fpr_val, tpr_val, label=f'ROC Validación (AUC = {roc_auc_val:.4f})', color='green')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Matriz de Confusión (Prueba)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión (Prueba, umbral=0.5)")
    plt.colorbar()
    tick_marks = range(2)
    plt.xticks(tick_marks, ["No Defecto", "Defecto"])
    plt.yticks(tick_marks, ["No Defecto", "Defecto"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    plt.show()
    # === Permutation Feature Importance ===
    print("\n🔍 Calculando Permutation Feature Importance...")

    # Obtener nombres de las características del preprocesador
    feature_names = []
    for name, transformer, columns in best_pipeline.named_steps['preprocessor'].transformers_:
        if name == 'num':
            select_kbest = transformer.named_steps['select']
            mask = select_kbest.get_support()  # características seleccionadas
            selected_features = [columns[i] for i in range(len(columns)) if mask[i]]
            feature_names.extend(selected_features)
        elif name == 'text':
            n_components = transformer.named_steps['svd'].n_components
            feature_names.extend([f'text_component_{i+1}' for i in range(n_components)])
        elif name == 'cat':
            feature_names.extend(transformer.get_feature_names_out(columns))

    # Calcular la importancia por permutación
    result = permutation_importance(
        best_pipeline,
        X_test_feats,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring='f1_macro'
    )

    # Asegurar longitud compatible
    n_features = len(result.importances_mean)
    importancia_perm = pd.DataFrame({
        'Feature': feature_names[:n_features],
        'Importance Mean': result.importances_mean,
        'Importance Std': result.importances_std
    }).sort_values(by='Importance Mean', ascending=False)

    # Mostrar top 10
    print("\n📊 Top 10 características más importantes (Permutation Importance):")
    print(importancia_perm.head(10).to_string(index=False))

    # Visualizar las 20 más importantes
    plt.figure(figsize=(10, 6))
    plt.barh(importancia_perm['Feature'].head(20)[::-1],
            importancia_perm['Importance Mean'].head(20)[::-1],
            xerr=importancia_perm['Importance Std'].head(20)[::-1],
            color='steelblue')
    plt.xlabel("Importancia promedio (f1_macro)")
    plt.title("Top 20 características por Permutation Feature Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Obtener importancias promediadas de todos los clasificadores calibrados
    #importances = np.mean([
    #    clf.estimator.feature_importances_  # Usar 'estimator' en lugar de 'base_estimator'
    #    for clf in best_pipeline.named_steps['clf'].calibrated_classifiers_
    #], axis=0)

    # Combinar nombres e importancias
    #feature_importance = pd.DataFrame({
    #    'Feature': feature_names,
    #    'Importance': importances
    #}).sort_values(by='Importance', ascending=False)

    # Mostrar las 10 características más importantes
    #print("\nTop 10 características más importantes:")
    #print(feature_importance.head(10).to_string(index=False))

    # Opcional: Guardar en archivo
    #feature_importance.to_csv('feature_importance.csv', index=False)

    # --- Métricas por Repositorio ---
    print("\n=== Métricas por Repositorio (Prueba) ===")
    df_test['pred'] = y_pred

    def get_f1_score(group):
        repo_name = group['repo'].iloc[0]
        try:
            if 1 not in group['label'].values and 1 not in group['pred'].values:
                print(f"Repositorio '{repo_name}' sin clase '1' (muestras: {len(group)})")
                return 0
            report = classification_report(group['label'], group['pred'], output_dict=True, zero_division=0)
            return report.get('1', {'f1-score': 0})['f1-score']
        except ValueError as e:
            print(f"Error en repositorio '{repo_name}': {e} (muestras: {len(group)})")
            return 0

    repo_metrics = df_test.groupby('repo').apply(get_f1_score).sort_values(ascending=False)
    print("F1-score para clase 1 por repositorio:")
    print(repo_metrics.to_string())

    # --- Ejemplos de Errores de Predicción ---
    print("\n=== Ejemplos de Errores de Predicción (Prueba) ===")
    fp = df_test[(df_test['label'] == 0) & (df_test['pred'] == 1)][['repo', 'filepath', 'content']].head(3)
    fn = df_test[(df_test['label'] == 1) & (df_test['pred'] == 0)][['repo', 'filepath', 'content']].head(3)
    fp['content'] = fp['content'].str[:200]
    fn['content'] = fn['content'].str[:200]
    print("\nFalsos Positivos (predicho como defecto, pero no lo es):")
    print(fp.to_string())
    print("\nFalsos Negativos (defecto no detectado):")
    print(fn.to_string())

    # --- Guardar Modelo ---
    joblib.dump(pipeline, "rf_pipeline_model.joblib")
    print("\n✅ Modelo guardado en 'rf_pipeline_model.joblib'.")

    # ==============================
    # Análisis de Repositorios
    # ==============================
    print("\n=== Análisis de Repositorios y Nomenclatura de Archivos ===")

    # Estadísticas de Repositorios
    repo_stats = df_train.groupby('repo').agg(
        n_commits=('repo', 'count'),
        n_unique_files=('filepath', 'nunique')
    ).reset_index()
    repo_stats = repo_stats.merge(repo_metrics.rename('f1_score'), on='repo', how='left').fillna({'f1_score': 0})
    print("\nTop 10 repositorios por cantidad de commits con F1-score:")
    print(repo_stats.sort_values(by='n_commits', ascending=False)[['repo', 'n_commits', 'n_unique_files', 'f1_score']].head(10).to_string(index=False))

    # Distribución de Extensiones
    df_train['file_extension'] = df_train['filepath'].apply(lambda x: x.split('.')[-1] if '.' in x else 'none')
    extension_stats = df_train.groupby('repo')['file_extension'].value_counts(normalize=True).unstack().fillna(0)
    print("\nDistribución de extensiones de archivo en algunos repositorios:")
    print(extension_stats.head(10).to_string())

    # Longitud de Nombres de Archivos
    file_naming_stats = df_train.groupby('repo').agg(
        avg_filepath_length=('filepath', lambda x: x.str.len().mean())
    ).reset_index()
    print("\nLongitud promedio de nombres de archivos por repositorio (top 10):")
    print(file_naming_stats.sort_values(by='avg_filepath_length', ascending=False).head(10).to_string(index=False))

    # Visualización: Commits vs. F1-score
    plt.figure(figsize=(8, 6))
    plt.scatter(repo_stats['n_commits'], repo_stats['f1_score'], alpha=0.5)
    plt.xlabel("Número de Commits")
    plt.ylabel("F1-score Clase '1'")
    plt.title("Commits vs. F1-score por Repositorio")
    plt.grid(True)
    plt.show()

    print("\n✅ Análisis de repositorios completado.")
    print("\n🎉 Proceso completado.")