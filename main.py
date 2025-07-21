import time
import os
import argparse
from datetime import datetime
from config import GENERAL_PARAMS, DATASET_SIZES
from train import load_data, process_data, extract_features, train_and_evaluate
from pipeline import create_preprocessor, create_pipelines

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Defect Prediction Model: Entrenamiento y evaluación de modelos de predicción de defectos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=GENERAL_PARAMS['RANDOM_STATE'],
        help="Semilla para reproducibilidad (entero positivo)"
    )
    parser.add_argument(
        "--data_type",
        type=int,
        choices=[0, 1],
        default=GENERAL_PARAMS['DATA_TYPE'],
        help="Tipo de datos: 0 (sin balancear) o 1 (balanceado)"
    )
    parser.add_argument(
        "--enable_hyperparam_search",
        type=lambda s: s.lower() == 'true',
        default=GENERAL_PARAMS['ENABLE_HYPERPARAM_SEARCH'],
        help="Activar/desactivar búsqueda de hiperparámetros (true/false)"
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=DATASET_SIZES['DATA_FRACTION'],
        help="Fracción de datos a usar para entrenamiento (entre 0 y 1)"
    )
    args = parser.parse_args()

    # Validaciones
    if args.random_state < 0:
        parser.error("random_state debe ser un entero positivo")
    if not (0 <= args.data_fraction <= 1):
        parser.error("data_fraction debe estar entre 0 y 1")

    return args


if __name__ == "__main__":
    start = time.time()

    args = parse_arguments()
    
    GENERAL_PARAMS['RANDOM_STATE'] = args.random_state
    GENERAL_PARAMS['DATA_TYPE'] = args.data_type
    GENERAL_PARAMS['ENABLE_HYPERPARAM_SEARCH'] = args.enable_hyperparam_search
    DATASET_SIZES['DATA_FRACTION'] = args.data_fraction
    
    print("\n=== Parámetros de ejecución ===")
    print(f"RANDOM_STATE: {GENERAL_PARAMS['RANDOM_STATE']}")
    print(f"DATA_TYPE: {GENERAL_PARAMS['DATA_TYPE']} (0: sin balancear, 1: balanceado)")
    print(f"ENABLE_HYPERPARAM_SEARCH: {GENERAL_PARAMS['ENABLE_HYPERPARAM_SEARCH']}")
    print(f"DATA_FRACTION: {DATASET_SIZES['DATA_FRACTION']}")
    print("=============================\n")
    
    # Crear carpeta general "ejecuciones" si no existe
    general_dir = "ejecuciones"
    if not os.path.exists(general_dir):
        os.makedirs(general_dir)
    
    # Generar nombre único para la subcarpeta basado en la fecha y hora
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subdir = os.path.join(general_dir, timestamp)
    os.makedirs(subdir)
    
    numeric_features = [
        "loc_full", "num_defs_full", "num_classes_full", "num_imports_full", "has_try_except_full",
        "has_if_else_full", "uses_torch_full", "uses_numpy_full", "uses_cv2_full", "has_return_full",
        "has_raise_full", "length_full", "avg_line_length_full", "code_density_full", "token_entropy_full",
        "num_indent_levels_full", "has_unclosed_parens_full", "repetition_score_full", "num_decision_points_full",
        "num_external_calls_full", "num_operators_full", "avg_words_per_line_full", "num_docstrings_full",
        "num_error_keywords_full", "loc_diff", "num_defs_diff", "num_classes_diff", "num_imports_diff",
        "has_try_except_diff", "has_if_else_diff", "uses_torch_diff", "uses_numpy_diff", "uses_cv2_diff",
        "has_return_diff", "has_raise_diff", "length_diff", "avg_line_length_diff", "code_density_diff",
        "token_entropy_diff", "num_indent_levels_diff", "has_unclosed_parens_diff", "repetition_score_diff",
        "num_decision_points_diff", "num_external_calls_diff", "num_operators_diff", "avg_words_per_line_diff",
        "num_docstrings_diff", "num_error_keywords_diff", "commit_hour", "commit_day", "filepath_length",
        "file_change_freq"
    ]
    categorical_features = ['repo_domain']

    dfs_line, dfs_jit = load_data()
    
    X_train, y_train, X_test, y_test, X_val, y_val, df_train, df_test, df_val = process_data(
        dfs_line, dfs_jit, GENERAL_PARAMS['DATA_TYPE']
    )
    
    X_train_feats, X_test_feats, X_val_feats, numeric_features, categorical_features = extract_features(
        X_train, X_test, X_val, numeric_features, categorical_features
    )
    
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    pipelines = create_pipelines(preprocessor)
    
    train_and_evaluate(
        pipelines, X_train_feats, y_train, X_test_feats, y_test, X_val_feats, y_val,
        df_train, df_test, df_val, numeric_features, categorical_features, output_dir=subdir
    )
    
    print("\n🎉 Proceso completado.")
    end = time.time()
    print(f"Tiempo total de ejecución: {end - start:.2f} segundos")