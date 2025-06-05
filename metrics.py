from collections import Counter
import math

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
    keywords = ['if ', 'elif ', 'for ', 'while ', 'try:', 'except', ' and ', ' or ']
    lines = content.splitlines()
    return sum(any(kw in line for kw in keywords) for line in lines if line.strip())

def num_external_calls(content):
    lines = content.splitlines()
    loc = len(lines) or 1
    return content.count('.') / loc

def map_repo_to_domain(repo):
    mapping = {
        "lightning|jax|transformers|ray|scikit-learn|spacy|yolov5": "machine_learning",
        "ansible|airflow": "automation",
        "celery": "task_queue",
        "openpilot": "autonomous_driving",
        "django|django-rest-framework": "web_framework",
        "freqtrade": "trading",
        "core": "iot",
        "numpy|pandas|redash": "data_science",
        "pipenv|poetry|black": "development",
        "scrapy": "web_scraping",
        "sentry": "logging"
    }
    repo_lower = repo.lower()
    for key, value in mapping.items():
        if any(k in repo_lower for k in key.split('|')):
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
        }