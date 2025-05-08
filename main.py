import pandas as pd
import time
import math
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# Scikit-learn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import SelectKBest, f_classif  # Descomentar para selección de características
# from imblearn.over_sampling import SMOTE  # Descomentar para SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline

# ==============================
# Configuración de Parámetros
# ==============================

# Parámetros Generales
GENERAL_PARAMS = {
    'RANDOM_STATE': 42,
    'DATA_FRACTION': 1.0
}

# Parámetros para TfidfVectorizer
TFIDF_PARAMS = {
    'max_features': 20000,
    'ngram_range': (1, 2),
    'min_df': 1,
    'max_df': 0.8,
    'stop_words': ['def', 'import', 'class'],
    'lowercase': True,
    'analyzer': 'word',
    'norm': 'l2',
    'sublinear_tf': False
    # Sugerencia: 'min_df': 2, 'stop_words': ['def', 'import', 'class', 'if', 'for', 'while', 'return', 'try', 'except'], 'sublinear_tf': True
}

# Parámetros para TruncatedSVD
SVD_PARAMS = {
    'n_components': 20,
    'algorithm': 'randomized',
    'n_iter': 10,
    'random_state': GENERAL_PARAMS['RANDOM_STATE']
    # Sugerencia: 'n_components': 50
}

# Parámetros para RandomForestClassifier
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'class_weight': 'balanced',
    'random_state': GENERAL_PARAMS['RANDOM_STATE']
    # Sugerencia: 'n_estimators': 200, 'max_depth': 30, 'max_features': 'sqrt'
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
    return content.count('.')

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
    return sum(content_lower.count(word) for word in error_keywords)

def extraer_metricas(content):
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
        "todo_ratio": todo_ratio
    }

def extraer_features(df_input):
    print("Extrayendo características de un batch de datos...")
    try:
        df_input['content'] = df_input['content'].apply(
            lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x))
        df_input['commit_datetime'] = pd.to_datetime(df_input['datetime'], errors='coerce')
        df_input['commit_hour'] = df_input['commit_datetime'].dt.hour
        df_input['commit_day'] = df_input['commit_datetime'].dt.dayofweek
        df_input['num_methods'] = df_input['methods'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df_input['filepath_length'] = df_input['filepath'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df_input['file_purpose'] = df_input['filepath'].apply(map_path_to_purpose)
        df_input['file_change_freq'] = df_input['filepath'].map(df_input['filepath'].value_counts())
        
        metricas = df_input['content'].apply(extraer_metricas).apply(pd.Series)
        df_features = pd.concat([
            metricas, 
            df_input[['commit_hour', 'commit_day', 'num_methods', 'filepath_length', 'repo', 'file_purpose', 'file_change_freq']]
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
if __name__ == "__main__":
    start = time.time()
    
    print("Cargando los datasets...")
    df_train = pd.read_parquet("jit_bug_prediction_splits/random/train.parquet.gzip")
    df_test = pd.read_parquet("jit_bug_prediction_splits/random/test.parquet.gzip")
    df_val = pd.read_parquet("jit_bug_prediction_splits/random/val.parquet.gzip")
    print("Datasets cargados.")

    df_train = df_train.sample(frac=GENERAL_PARAMS['DATA_FRACTION'], random_state=GENERAL_PARAMS['RANDOM_STATE'])
    df_test = df_test.sample(frac=GENERAL_PARAMS['DATA_FRACTION'], random_state=GENERAL_PARAMS['RANDOM_STATE'])
    df_val = df_val.sample(frac=GENERAL_PARAMS['DATA_FRACTION'], random_state=GENERAL_PARAMS['RANDOM_STATE'])
    
    df_train['label'] = df_train['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
    df_test['label'] = df_test['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
    df_val['label'] = df_val['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
    
    df_train['content'] = df_train['content'].apply(lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x))
    df_test['content'] = df_test['content'].apply(lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x))
    df_val['content'] = df_val['content'].apply(lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x))
    
    X_train = df_train
    X_test = df_test
    X_val = df_val
    y_train = df_train['label']
    y_test = df_test['label']
    y_val = df_val['label']
    print("Conjuntos de datos preparados.")
    
    print("Extrayendo características para el conjunto de entrenamiento...")
    X_train_feats = extraer_features(X_train)
    if X_train_feats is None:
        exit(1)
    print("Extrayendo características para el conjunto de prueba...")
    X_test_feats = extraer_features(X_test)
    if X_test_feats is None:
        exit(1)
    print("Extrayendo características para el conjunto de validación...")
    X_val_feats = extraer_features(X_val)
    if X_val_feats is None:
        exit(1)
    
    numeric_features = [
        "loc", "num_defs", "num_classes", "num_imports", "has_try_except", "has_if_else",
        "uses_torch", "uses_numpy", "uses_cv2", "has_return", "has_raise", "num_comments",
        "num_empty_lines", "length", "avg_line_length", "comment_ratio", "empty_ratio",
        "code_density", "token_entropy", "num_indent_levels", "has_tab_mix", "has_unclosed_parens",
        "repetition_score", "num_decision_points", "num_external_calls",
        "commit_hour", "commit_day", "num_methods", "filepath_length", "file_change_freq",
        "avg_indent_length", "max_indent_length", "num_todo_comments", "num_operators",
        "avg_words_per_line", "num_docstrings", "num_error_keywords", "todo_ratio"
    ]
    
    print("Definiendo el preprocesador de datos...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SkPipeline(steps=[
                ('imputer', SimpleImputer(**IMPUTER_PARAMS)),
                ('scaler', StandardScaler(**SCALER_PARAMS))
                # Descomentar para selección de características:
                # , ('select', SelectKBest(f_classif, k=20))
            ]), numeric_features),
            ('text', SkPipeline(steps=[
                ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
                ('svd', TruncatedSVD(**SVD_PARAMS))
            ]), 'content_text'),
            ('cat', OneHotEncoder(**OHE_PARAMS), ['repo_domain', 'file_purpose'])
        ]
    )
    
    print("Construyendo el pipeline completo con Random Forest...")
    # Descomentar para SMOTE:
    # pipeline = ImbPipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('smote', SMOTE(random_state=GENERAL_PARAMS['RANDOM_STATE'])),
    #     ('clf', RandomForestClassifier(**RF_PARAMS))
    # ])
    pipeline = SkPipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(**RF_PARAMS))
    ])
    
    print("Entrenando el modelo...")
    pipeline.fit(X_train_feats, y_train)
    end = time.time()
    print(f"\n🕒 Tiempo de entrenamiento: {end - start:.2f} segundos")
    
    print("Realizando predicciones en el conjunto de prueba...")
    start_pred = time.time()
    y_pred = pipeline.predict(X_test_feats)
    y_prob = pipeline.predict_proba(X_test_feats)[:, 1]
    end_pred = time.time()
    print(f"🕒 Tiempo de predicción: {end_pred - start_pred:.2f} segundos")
    
    print("\n📊 Reporte de clasificación (test, umbral=0.5):")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión (test, umbral=0.5):")
    print(cm)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Área bajo la curva ROC (test): {roc_auc:.4f}")
    
    # Probar diferentes umbrales
    print("\nProbando diferentes umbrales para la clase 1...")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        print(f"\nUmbral: {thresh}")
        print(classification_report(y_test, y_pred))
        print("Matriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))
    
    print("\nRealizando predicciones en el conjunto de validación...")
    y_val_pred = pipeline.predict(X_val_feats)
    y_val_prob = pipeline.predict_proba(X_val_feats)[:, 1]
    print("\n📊 Reporte de clasificación (validación, umbral=0.5):")
    print(classification_report(y_val, y_val_pred))
    
    cm_val = confusion_matrix(y_val, y_val_pred)
    print("Matriz de Confusión (validación, umbral=0.5):")
    print(cm_val)
    
    fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_prob)
    roc_auc_val = roc_auc_score(y_val, y_val_prob)
    print(f"Área bajo la curva ROC (validación): {roc_auc_val:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Test (AUC = {roc_auc:.4f})')
    plt.plot(fpr_val, tpr_val, label=f'ROC Val (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión (test, umbral=0.5)")
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
    
    joblib.dump(pipeline, "rf_pipeline_model.joblib")
    print("Modelo guardado en 'rf_pipeline_model.joblib'.")
    
    print("\n=== Análisis de Repositorios y Nomenclatura de Archivos ===")
    repo_stats = df_train.groupby('repo').agg(
        n_commits=('repo', 'count'),
        n_unique_files=('filepath', 'nunique')
    ).reset_index()
    
    print("\nTop 10 repositorios por cantidad de commits:")
    print(repo_stats.sort_values(by='n_commits', ascending=False).head(10))
    
    df_train['file_extension'] = df_train['filepath'].apply(lambda x: x.split('.')[-1] if '.' in x else 'none')
    extension_stats = df_train.groupby('repo')['file_extension'].value_counts(normalize=True).unstack().fillna(0)
    
    print("\nDistribución de extensiones de archivo en algunos repositorios:")
    print(extension_stats.head(10))
    
    file_naming_stats = df_train.groupby('repo').agg(
        avg_filepath_length=('filepath', lambda x: x.str.len().mean())
    ).reset_index()
    
    print("\nLongitud promedio de nombres de archivos por repositorio (top 10 con mayor longitud):")
    print(file_naming_stats.sort_values(by='avg_filepath_length', ascending=False).head(10))
    
    print("\nAnálisis de repositorios completado.")
    print("\nProceso completado.")