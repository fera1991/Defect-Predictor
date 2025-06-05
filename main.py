import time
from config import GENERAL_PARAMS
from train import load_data, process_data, extract_features, train_and_evaluate
from pipeline import create_preprocessor, create_pipelines

if __name__ == "__main__":
    start = time.time()
    
    numeric_features = [
        "loc_full", "num_defs_full", "num_classes_full", "num_imports_full", "has_try_except_full",
        "has_if_else_full", "uses_torch_full", "uses_numpy_full", "uses_cv2_full", "has_return_full",
        "has_raise_full", "num_comments_full", "num_empty_lines_full", "length_full",
        "avg_line_length_full", "comment_ratio_full", "empty_ratio_full", "code_density_full",
        "token_entropy_full", "num_indent_levels_full", "has_tab_mix_full", "has_unclosed_parens_full",
        "repetition_score_full", "num_decision_points_full", "num_external_calls_full",
        "avg_indent_length_full", "max_indent_length_full", "num_todo_comments_full",
        "num_operators_full", "avg_words_per_line_full", "num_docstrings_full",
        "num_error_keywords_full", "todo_ratio_full",
        "loc_diff", "num_defs_diff", "num_classes_diff", "num_imports_diff", "has_try_except_diff",
        "has_if_else_diff", "uses_torch_diff", "uses_numpy_diff", "uses_cv2_diff", "has_return_diff",
        "has_raise_diff", "num_comments_diff", "num_empty_lines_diff", "length_diff",
        "avg_line_length_diff", "comment_ratio_diff", "empty_ratio_diff", "code_density_diff",
        "token_entropy_diff", "num_indent_levels_diff", "has_tab_mix_diff", "has_unclosed_parens_diff",
        "repetition_score_diff", "num_decision_points_diff", "num_external_calls_diff",
        "avg_indent_length_diff", "max_indent_length_diff", "num_todo_comments_diff",
        "num_operators_diff", "avg_words_per_line_diff", "num_docstrings_diff",
        "num_error_keywords_diff", "todo_ratio_diff",
        "commit_hour", "commit_day", "num_methods", "filepath_length", "file_change_freq"
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
        df_train, df_test, df_val, numeric_features, categorical_features
    )
    
    print("\n🎉 Proceso completado.")
    end = time.time()
    print(f"Tiempo total de ejecución: {end - start:.2f} segundos")