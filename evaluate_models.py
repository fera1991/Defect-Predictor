import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted
from features import extraer_features
from utils import encontrar_umbral_optimo_mcc
from config import GENERAL_PARAMS, DATASET_SIZES
import logging
from datetime import datetime
from sklearn.utils import resample
import argparse
import sys

# Configurar logging
logging.basicConfig(
    filename='evaluate.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Evaluación de modelos preentrenados para predicción de defectos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directorio donde se encuentran los modelos preentrenados (obligatorio)"
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
        "--fixed_threshold",
        type=float,
        default=None,
        help="Umbral fijo para clasificación (entre 0 y 1, opcional)"
    )
    parser.add_argument(
        "--val_percent",
        type=float,
        default=0.2,
        help="Porcentaje del conjunto de validación a usar (entre 0 y 1)"
    )
    parser.add_argument(
        "--test_percent",
        type=float,
        default=0.2,
        help="Porcentaje del conjunto de prueba a usar (entre 0 y 1)"
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        # Mostrar mensaje solo si falta --models_dir
        print("\n=== NOTA IMPORTANTE ===")
        print("Debe especificar la ubicación del directorio de modelos preentrenados usando el argumento --models_dir.")
        print("Ejemplo: python evaluate_models.py --models_dir ejecuciones/modelos_finales")
        print("Use --help para más información sobre los parámetros disponibles.")
        print("======================\n")
        sys.exit(2)  # Salir con el código de error estándar de argparse

    # Validaciones
    if args.random_state < 0:
        parser.error("random_state debe ser un entero positivo")
    if args.fixed_threshold is not None and not (0 <= args.fixed_threshold <= 1):
        parser.error("fixed_threshold debe estar entre 0 y 1")
    if not (0 <= args.val_percent <= 1):
        parser.error("val_percent debe estar entre 0 y 1")
    if not (0 <= args.test_percent <= 1):
        parser.error("test_percent debe estar entre 0 y 1")
    if not os.path.isdir(args.models_dir):
        logging.error(f"El directorio de modelos '{args.models_dir}' no existe.")
        print(f"❌ Error: El directorio de modelos '{args.models_dir}' no existe. Proporcione un directorio válido.")
        sys.exit(1)

    return args

def load_data():
    logging.info("Cargando los datasets...")
    print("Cargando los datasets...")
    df_test_line = pd.read_parquet("line_bug_prediction_splits/random/test.parquet.gzip")
    df_val_line = pd.read_parquet("line_bug_prediction_splits/random/val.parquet.gzip")
    df_test_jit = pd.read_parquet("jit_bug_prediction_splits/random/test.parquet.gzip")
    df_val_jit = pd.read_parquet("jit_bug_prediction_splits/random/val.parquet.gzip")
    logging.info(f"Columnas en df_test_line: {df_test_line.columns.tolist()}")
    logging.info(f"Columnas en df_test_jit: {df_test_jit.columns.tolist()}")
    print("Columnas en df_test_line:", df_test_line.columns.tolist())
    print("Columnas en df_test_jit:", df_test_jit.columns.tolist())
    logging.info("Datasets cargados.")
    print("Datasets cargados.")
    return df_test_line, df_val_line, df_test_jit, df_val_jit

def process_data(df_test_line, df_val_line, df_test_jit, df_val_jit, data_type=GENERAL_PARAMS['DATA_TYPE']):
    logging.info("Fusionando datasets JIT y Line...")
    print("\nFusionando datasets JIT y Line...")
    df_test = df_test_jit.merge(df_test_line[['commit', 'repo', 'filepath', 'content']], 
                               on=['commit', 'repo', 'filepath'], 
                               suffixes=('_diff', '_full'))
    df_val = df_val_jit.merge(df_val_line[['commit', 'repo', 'filepath', 'content']], 
                             on=['commit', 'repo', 'filepath'], 
                             suffixes=('_diff', '_full'))
    
    logging.info("Asignando etiquetas...")
    print("\nAsignando etiquetas...")
    df_test['label'] = df_test['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
    df_val['label'] = df_val['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
    
    logging.info("Distribución de clases original:")
    logging.info(f"Prueba ({len(df_test)} ejemplos): {df_test['label'].value_counts(normalize=True).to_dict()}")
    logging.info(f"Validación ({len(df_val)} ejemplos): {df_val['label'].value_counts(normalize=True).to_dict()}")
    print("\nDistribución de clases original:")
    print(f"Prueba ({len(df_test)} ejemplos):", df_test['label'].value_counts(normalize=True))
    print(f"Validación ({len(df_val)} ejemplos):", df_val['label'].value_counts(normalize=True))
    
    if data_type == 0:
        logging.info("Tipo 0: Usar datasets originales sin modificar ni estratificar")
        print("\nTipo 0: Usar datasets originales sin modificar ni estratificar")
    elif data_type == 1:
        logging.info("Tipo 1: Balancear val, mantener test balanceado con proporción específica")
        print("\nTipo 1: Balancear val, mantener test balanceado con proporción específica")
        
        TRAIN_VAL_PERCENT = DATASET_SIZES['TRAIN_VAL_PERCENT']
        TRAIN_TEST_PERCENT = DATASET_SIZES['TRAIN_TEST_PERCENT']
        TEST_CLASS1_RATIO = DATASET_SIZES['TEST_CLASS1_RATIO']
        DATA_FRACTION = DATASET_SIZES['DATA_FRACTION']
        
        total_train_size = 193420  # Tamaño total del conjunto de entrenamiento original
        
        logging.info("Ajustando conjunto de prueba...")
        print("\nAjustando conjunto de prueba...")
        target_test_size = int(total_train_size * TRAIN_TEST_PERCENT)
        test_0 = df_test[df_test['label'] == 0]
        test_1 = df_test[df_test['label'] == 1]
        target_class1_size = int(target_test_size * TEST_CLASS1_RATIO)
        target_class0_size = target_test_size - target_class1_size
        
        test_1_resampled = resample(
            test_1,
            n_samples=target_class1_size,
            replace=len(test_1) < target_class1_size,
            random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        
        test_0_resampled = resample(
            test_0,
            n_samples=target_class0_size,
            replace=len(test_0) < target_class0_size,
            random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        
        df_test = pd.concat([test_0_resampled, test_1_resampled])
        
        logging.info("Ajustando conjunto de validación...")
        print("\nAjustando conjunto de validación...")
        target_val_size = int(total_train_size * TRAIN_VAL_PERCENT)
        val_0 = df_val[df_val['label'] == 0]
        val_1 = df_val[df_val['label'] == 1]
        target_val_class_size = target_val_size // 2
        
        val_1_resampled = resample(
            val_1,
            n_samples=target_val_class_size,
            replace=len(val_1) < target_val_class_size,
            random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        
        val_0_resampled = resample(
            val_0,
            n_samples=target_val_class_size,
            replace=len(val_0) < target_val_class_size,
            random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        
        df_val_balanced = pd.concat([val_0_resampled, val_1_resampled])
        df_val = df_val_balanced.sample(frac=DATA_FRACTION, random_state=GENERAL_PARAMS['RANDOM_STATE'])
        
        if abs(len(df_test) - target_test_size) > 10 or abs(df_test['label'].mean() - TEST_CLASS1_RATIO) > 0.01:
            logging.error(f"Conjunto de prueba no tiene ~{target_test_size} ejemplos o proporción incorrecta")
            raise ValueError(f"El conjunto de prueba no tiene ~{target_test_size} ejemplos o proporción incorrecta (esperado ~{TEST_CLASS1_RATIO*100}% clase 1)")
        if abs(df_val['label'].mean() - 0.5) > 0.01:
            logging.error("Conjunto de validación no está balanceado")
            raise ValueError("El conjunto de validación no está balanceado (~50%/50%)")
        
        logging.info("Distribución de clases final:")
        logging.info(f"Prueba ({len(df_test)} ejemplos): {df_test['label'].value_counts(normalize=True).to_dict()}")
        logging.info(f"Validación ({len(df_val)} ejemplos): {df_val['label'].value_counts(normalize=True).to_dict()}")
        print("\nDistribución de clases final:")
        print(f"Prueba ({len(df_test)} ejemplos):", df_test['label'].value_counts(normalize=True))
        print(f"Validación ({len(df_val)} ejemplos):", df_val['label'].value_counts(normalize=True))
    else:
        logging.error("data_type debe ser 0 o 1")
        raise ValueError("data_type debe ser 0 o 1")

    for df in [df_test, df_val]:
        df['content_diff'] = df['content_diff'].apply(
            lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x)
        ).str.replace('\\n', '\n')
        df['content_full'] = df['content_full'].apply(
            lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x)
        ).str.replace('\\n', '\n')

    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']
    X_val = df_val.drop(columns=['label'])
    y_val = df_val['label']

    logging.info(f"Tamaño del conjunto de prueba: {len(df_test)}")
    logging.info(f"Tamaño del conjunto de validación: {len(df_val)}")
    print(f"Tamaño del conjunto de prueba: {len(df_test)}")
    print(f"Tamaño del conjunto de validación: {len(df_val)}")
    logging.info("Conjuntos de datos preparados.")
    print("Conjuntos de datos preparados.")
    
    return X_test, y_test, X_val, y_val, df_test, df_val

def extract_features(X_test, X_val, numeric_features, categorical_features):
    logging.info("Extrayendo características para el conjunto de prueba...")
    print("\nExtrayendo características para el conjunto de prueba...")
    start_extract = time.time()
    X_test_feats = extraer_features(X_test)
    if X_test_feats is None:
        logging.error("No se encontraron características para el conjunto de prueba.")
        print("Error: No se encontraron características para el conjunto de prueba.")
        exit(1)
    end_extract = time.time()
    logging.info(f"Tiempo de extracción: {end_extract - start_extract:.2f} segundos")
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")

    logging.info("Extrayendo características para el conjunto de validación...")
    print("\nExtrayendo características para el conjunto de validación...")
    start_extract = time.time()
    X_val_feats = extraer_features(X_val)
    if X_val_feats is None:
        logging.error("No se pudieron extraer características para el conjunto de validación.")
        print("Error: No se pudieron extraer características para el conjunto de validación.")
        exit(1)
    end_extract = time.time()
    logging.info(f"Tiempo de extracción: {end_extract - start_extract:.2f} segundos")
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")

    logging.info(f"Número de características extraídas: {len(X_test_feats.columns)}")
    print(f"\nNúmero de características extraídas: {len(X_test_feats.columns)}")
    logging.info(f"Valores faltantes por característica:\n{X_test_feats.isna().sum().to_dict()}")
    print(f"\nValores faltantes por característica:\n{X_test_feats.isna().sum()}")

    constant_features = X_test_feats.columns[X_test_feats.nunique() == 1]
    logging.info(f"Características constantes: {constant_features.tolist()}")
    print("\nCaracterísticas constantes:", constant_features)
    if len(constant_features) > 0:
        X_test_feats = X_test_feats.drop(columns=constant_features)
        X_val_feats = X_val_feats.drop(columns=constant_features)
        numeric_features = [f for f in numeric_features if f not in constant_features]
        categorical_features = [f for f in categorical_features if f not in constant_features]
        logging.info(f"Características constantes eliminadas: {len(constant_features)}")
        print(f"Características constantes eliminadas: {len(constant_features)}")

    logging.info(f"Columnas en X_test_feats después de eliminar constantes: {X_test_feats.columns.tolist()}")
    print("\nColumnas en X_test_feats después de eliminar constantes:", X_test_feats.columns.tolist())
    logging.info(f"Características categóricas actualizadas: {categorical_features}")
    print("Características categóricas actualizadas:", categorical_features)

    return X_test_feats, X_val_feats, numeric_features, categorical_features

def evaluate_models(models_dir, X_test_feats, y_test, X_val_feats, y_val, numeric_features, categorical_features, output_dir, fixed_threshold=None):
    logging.info(f"Resultados se guardarán en '{output_dir}'.")
    print(f"Resultados se guardarán en '{output_dir}'.")
    
    # Diccionario para mapear nombres de modelos a versiones más amigables
    model_name_mapping = {
        'Randomforest': 'Random Forest',
        'Xgboost': 'XGBoost',
        'Votingclassifier': 'Voting Classifier'
    }
    
    models = {}
    roc_data = {'test': {}, 'val': {}}
    f1_scores = {'test': {}, 'val': {}}
    comparison_data = []
    optimal_threshold_data = []

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    for model_file in model_files:
        model_name = model_file.replace('_pipeline_model.joblib', '').capitalize()
        model_path = os.path.join(models_dir, model_file)
        logging.info(f"Cargando modelo {model_name} desde '{model_path}'...")
        print(f"\nCargando modelo {model_name} desde '{model_path}'...")
        models[model_name] = joblib.load(model_path)

    for model_name, model in models.items():
        display_name = model_name_mapping.get(model_name, model_name)
        logging.info(f"Evaluando {display_name}...")
        print(f"\n=== Evaluando {display_name} ===")
        start_eval = time.time()

        try:
            logging.info(f"Estado del preprocesador para {display_name}...")
            print(f"\n🔍 Estado del preprocesador para {display_name}...")
            preprocessor = model.named_steps['preprocessor']
            for name, transformer, columns in preprocessor.transformers_[:-1]:
                logging.info(f"Transformador {name}: {transformer}")
                print(f"Transformador {name}: {transformer}")
                try:
                    check_is_fitted(transformer)
                    logging.info(f"Transformador {name} está ajustado.")
                    print(f"Transformador {name} está ajustado.")
                except:
                    logging.warning(f"Transformador {name} NO está ajustado.")
                    print(f"Transformador {name} NO está ajustado.")
                
            logging.info(f"Generando nombres de características para {display_name}...")
            print(f"\n🔍 Generando nombres de características para {display_name}...")
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_[:-1]:
                try:
                    iname = name.replace("'", "")
                    if iname == 'num':
                        select_kbest = transformer.named_steps.get('select')
                        if select_kbest:
                            check_is_fitted(select_kbest)
                            mask = select_kbest.get_support()
                            selected_features = [columns[i] for i in range(len(columns)) if mask[i]]
                            feature_names.extend(selected_features)
                        else:
                            feature_names.extend(columns)
                    elif iname == 'cat':
                        check_is_fitted(transformer)
                        feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out())
                    elif iname in ['text_full', 'text_diff']:
                        n_components = transformer.named_steps['svd'].n_components
                        feature_names.extend([f'{iname}_component_{i+1}' for i in range(n_components)])
                except Exception as e:
                    logging.error(f"Error al procesar transformador {iname}: {e}")
                    print(f"Error al procesar transformador {iname}: {e}")
                    feature_names.extend(columns if iname in ['num', 'cat'] else [f'{iname}_component_{i+1}' for i in range(100)])

            for dataset_name, X, y in [('test', X_test_feats, y_test), ('val', X_val_feats, y_val)]:
                logging.info(f"Evaluación en Conjunto {dataset_name.capitalize()} ({display_name})...")
                print(f"\n=== Evaluación en Conjunto {dataset_name.capitalize()} ({display_name}) ===")
                start_pred = time.time()
                try:
                    y_prob = model.predict_proba(X)[:, 1]
                except AttributeError:
                    y_prob = model.decision_function(X) if hasattr(model, 'decision_function') else model.predict(X)
                end_pred = time.time()
                logging.info(f"Tiempo de predicción: {end_pred - start_pred:.2f} segundos")
                print(f"🕒 Tiempo de predicción: {end_pred - start_pred:.2f} segundos")

                # Calcular umbral óptimo basado en MCC
                umbral_mcc, mcc_val = encontrar_umbral_optimo_mcc(y, y_prob)
                logging.info(f"Umbral óptimo (MCC): {umbral_mcc:.3f}, MCC: {mcc_val:.3f}")
                print(f"🔍 Umbral óptimo (MCC): {umbral_mcc:.3f}, MCC: {mcc_val:.3f}")

                # Usar umbral fijo si se proporciona, de lo contrario usar umbral óptimo
                threshold = fixed_threshold if fixed_threshold is not None else umbral_mcc
                threshold_label = f"fijo={fixed_threshold:.3f}" if fixed_threshold is not None else f"óptimo MCC={umbral_mcc:.3f}"
                y_pred = (y_prob >= threshold).astype(int)

                logging.info(f"Reporte de Clasificación (umbral {threshold_label}, {display_name}):")
                logging.info(classification_report(y, y_pred))
                print(f"\n📊 Reporte de Clasificación (umbral {threshold_label}, {display_name}):")
                print(classification_report(y, y_pred))
                logging.info(f"Matriz de Confusión (umbral {threshold_label}, {display_name}):")
                logging.info(str(confusion_matrix(y, y_pred)))
                print(f"Matriz de Confusión (umbral {threshold_label}, {display_name}):")
                print(confusion_matrix(y, y_pred))

                report_dict = classification_report(y, y_pred, output_dict=True)
                accuracy = accuracy_score(y, y_pred)  # Calcular accuracy
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = roc_auc_score(y, y_prob)
                roc_data[dataset_name][model_name] = (fpr, tpr, roc_auc)
                f1_scores[dataset_name][model_name] = report_dict['1']['f1-score']
                logging.info(f"Área bajo la curva ROC: {roc_auc:.4f}")
                logging.info(f"Accuracy: {accuracy:.4f}")
                print(f"Área bajo la curva ROC: {roc_auc:.4f}")
                print(f"Accuracy: {accuracy:.4f}")

                for class_label in ['0', '1']:
                    comparison_data.append({
                        'Modelo': display_name,
                        'Conjunto': dataset_name.capitalize(),
                        'Clase': 'Sin Defectos' if class_label == '0' else 'Con Defectos',
                        'Precisión': report_dict[class_label]['precision'],
                        'Recall': report_dict[class_label]['recall'],
                        'F1-score': report_dict[class_label]['f1-score'],
                        'Soporte': report_dict[class_label]['support'],
                        'AUC-ROC': roc_auc,
                        'Accuracy': accuracy  # Agregar accuracy
                    })

                optimal_threshold_data.append({
                    'Modelo': display_name,
                    'Conjunto': dataset_name.capitalize(),
                    'Umbral': threshold,
                    'MCC': mcc_val if fixed_threshold is None else None,
                    'Precisión (Con Defectos)': report_dict['1']['precision'],
                    'Recall (Con Defectos)': report_dict['1']['recall'],
                    'F1-score (Con Defectos)': report_dict['1']['f1-score'],
                    'Accuracy': accuracy  # Agregar accuracy
                })

        except Exception as e:
            logging.error(f"Error inesperado en {display_name}: {e}")
            print(f"❌ Error inesperado en {display_name}: {e}")
            continue

    # Generar curvas ROC
    plt.figure(figsize=(10, 8))
    for model_name, (fpr, tpr, roc_auc) in roc_data['test'].items():
        plt.plot(fpr, tpr, label=f'{model_name_mapping.get(model_name, model_name)} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas ROC (Conjunto de Prueba)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_test.png"))
    plt.close()

    plt.figure(figsize=(10, 8))
    for model_name, (fpr_val, tpr_val, roc_auc_val) in roc_data['val'].items():
        plt.plot(fpr_val, tpr_val, label=f'{model_name_mapping.get(model_name, model_name)} (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas ROC (Conjunto de Validación)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_val.png"))
    plt.close()

    # Comparación de F1-scores
    logging.info("Comparación de F1-scores (Clase 1)...")
    print("\n=== Comparación de F1-scores (Clase 1) ===")
    f1_df = pd.DataFrame({
        'Test': f1_scores['test'],
        'Validation': f1_scores['val']
    })
    logging.info(f1_df.to_string())
    print(f1_df.to_string())
    f1_df.to_csv(os.path.join(output_dir, "f1_scores.csv"), index=True)

    # Comparación detallada
    comparison_df = pd.DataFrame(comparison_data)
    
    rf_present = 'Random Forest' in comparison_df['Modelo'].values
    xgb_present = 'XGBoost' in comparison_df['Modelo'].values
    if rf_present and xgb_present:
        logging.info("\nComparación Detallada de RandomForest y XGBoost...")
        print("\n=== Comparación Detallada de RandomForest y XGBoost ===")
        rf_xgb_comparison = comparison_df[comparison_df['Modelo'].isin(['Random Forest', 'XGBoost'])]
        logging.info(rf_xgb_comparison.to_string(index=False))
        print(rf_xgb_comparison.to_string(index=False))
        rf_xgb_comparison.to_csv(os.path.join(output_dir, "rf_xgb_comparison.csv"), index=False)
    else:
        logging.warning("No se puede realizar la comparación de RandomForest y XGBoost: uno o ambos modelos no están presentes.")
        print("⚠️ Advertencia: No se puede realizar la comparación de RandomForest y XGBoost porque uno o ambos modelos no están presentes.")

    logging.info("\nComparación Detallada de todos los modelos...")
    print("\n=== Comparación Detallada de todos los modelos ===")
    all_comparison = comparison_df
    logging.info(all_comparison.to_string(index=False))
    print(all_comparison.to_string(index=False))
    all_comparison.to_csv(os.path.join(output_dir, "all_models_comparison.csv"), index=False)

    # Comparación con umbral
    logging.info("\nComparación de Métricas con Umbral...")
    print("\n=== Comparación de Métricas con Umbral ===")
    optimal_df = pd.DataFrame(optimal_threshold_data)
    optimal_df = optimal_df.round(3)
    logging.info(optimal_df.to_string(index=False))
    print(optimal_df.to_string(index=False))
    optimal_df.to_csv(os.path.join(output_dir, "threshold_metrics.csv"), index=False)

    # Explicación del umbral
    logging.info("\nExplicación del Umbral:")
    logging.info("Por defecto, se utiliza el umbral óptimo basado en el coeficiente de correlación de Matthews (MCC) para maximizar el equilibrio entre precisión y recall.")
    logging.info("Si se proporciona un umbral fijo, se utiliza en lugar del óptimo. En la tabla 'threshold_metrics.csv', se observan las métricas correspondientes al umbral utilizado.")
    print("\n=== Explicación del Umbral ===")
    print("Por defecto, se utiliza el umbral óptimo basado en el coeficiente de correlación de Matthews (MCC) para maximizar el equilibrio entre precisión y recall.")
    print("Si se proporciona un umbral fijo, se utiliza en lugar del óptimo. En la tabla 'threshold_metrics.csv', se observan las métricas correspondientes al umbral utilizado.")

    logging.info("Evaluación completada.")
    print("\n🎉 Evaluación completada.")

if __name__ == "__main__":
    start = time.time()
    
    args = parse_arguments()
    
    # Actualizar parámetros
    GENERAL_PARAMS['RANDOM_STATE'] = args.random_state
    GENERAL_PARAMS['DATA_TYPE'] = args.data_type
    models_dir = args.models_dir
    val_percent = args.val_percent
    test_percent = args.test_percent
    fixed_threshold = args.fixed_threshold
    
    DATASET_SIZES['DATA_FRACTION'] = 1.0

    # Verificar que haya al menos un modelo en models_dir
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    if not model_files:
        logging.error(f"No se encontraron modelos en el directorio '{models_dir}'.")
        print(f"❌ Error: No se encontraron modelos en el directorio '{models_dir}'. Asegúrese de que contiene archivos .joblib.")
        sys.exit(1)

    general_dir = "evaluaciones"
    if not os.path.exists(general_dir):
        os.makedirs(general_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(general_dir, timestamp)
    os.makedirs(output_dir)
    
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

    df_test_line, df_val_line, df_test_jit, df_val_jit = load_data()
    X_test, y_test, X_val, y_val, df_test, df_val = process_data(df_test_line, df_val_line, df_test_jit, df_val_jit, data_type=1)
    X_test_feats, X_val_feats, numeric_features, categorical_features = extract_features(X_test, X_val, numeric_features, categorical_features)

    # Usar umbral óptimo por defecto (fixed_threshold=None)
    evaluate_models(models_dir, X_test_feats, y_test, X_val_feats, y_val, numeric_features, categorical_features, output_dir, fixed_threshold)
    
    end = time.time()
    logging.info(f"Tiempo total de ejecución: {end - start:.2f} segundos")
    print(f"\nTiempo total de ejecución: {end - start:.2f} segundos")