"""Allow-list projection: never export question/prompt/response/free-error text."""
from pathlib import Path
import pyarrow.compute as pc
import pyarrow.parquet as pq

CHAT_NUMERIC_COLUMNS = (
    'config_id', 'model', 'model_variant', 'generation_mode', 'language',
    'ideology', 'context_id', 'instr_type', 'reasoning_mode', 'instr_idx',
    'persona_class', 'persona_idx', 'key_type', 'perm_id', 'suffix_idx',
    'answer_mode', 'inference_backend', 'quantization', 'written_at', 'item_id',
    'question_id', 'question_page', 'candidate_keys', 'stage1_tokens',
    'stage1_finish_reason', 'classification_logprobs', 'classification_probs',
    'classification_pred_key', 'attempt',
)


def numeric_chat_table(table):
    missing = set(CHAT_NUMERIC_COLUMNS) - set(table.column_names)
    if missing:
        raise ValueError(f'Missing chat numeric fields: {sorted(missing)}')
    projected = table.select(CHAT_NUMERIC_COLUMNS).replace_schema_metadata(None)
    projected = projected.append_column('has_error', pc.is_valid(table['error']))
    return projected


def project_chat(source: Path, target: Path) -> int:
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    original = pq.read_table(source)
    numeric = numeric_chat_table(original)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(numeric, target, compression='zstd', compression_level=5)
    stored = pq.read_table(target)
    assert stored.equals(numeric)
    for column in CHAT_NUMERIC_COLUMNS:
        assert stored[column].equals(original[column]), column
    assert stored.schema.metadata is None
    return stored.num_rows
