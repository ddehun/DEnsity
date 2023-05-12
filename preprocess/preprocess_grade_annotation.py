import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os
from dataclasses import asdict


from evaluators.eval_utils import EvaluationExample


def _write_items_to_jsonl(data, fname):
    with open(fname, "w") as f:
        for el in data:
            f.write(json.dumps(el) + "\n")


def read_txt(fname):
    with open(fname, "r") as f:
        return [e.strip() for e in f.readlines()]


def preprocess_grade_dailydialogue(dirname, output_fname, skip_ranker=False, skip_generator=False):
    """Preprocessing DailyDialog++ dataset"""
    ranker_ctx_fname = os.path.join(dirname, "transformer_ranker/human_ctx.txt")
    ranker_hyp_fname = os.path.join(dirname, "transformer_ranker/human_hyp.txt")
    ranker_ref_fname = os.path.join(dirname, "transformer_ranker/human_ref.txt")
    ranker_score = os.path.join(dirname, "../human_score/dailydialog/transformer_ranker/human_score.txt")
    generator_ctx_fname = os.path.join(dirname, "transformer_generator/human_ctx.txt")
    generator_hyp_fname = os.path.join(dirname, "transformer_generator/human_hyp.txt")
    generator_ref_fname = os.path.join(dirname, "transformer_generator/human_ref.txt")
    generator_score = os.path.join(dirname, "../human_score/dailydialog/transformer_generator/human_score.txt")

    ctx, hyp, ref, score = list(
        map(
            read_txt,
            [ranker_ctx_fname, ranker_hyp_fname, ranker_ref_fname, ranker_score],
        )
    )
    assert len(ctx) == len(hyp) == len(ref) == len(score)
    generator_ctx, generator_hyp, generator_ref, generator_score = list(
        map(
            read_txt,
            [
                generator_ctx_fname,
                generator_hyp_fname,
                generator_ref_fname,
                generator_score,
            ],
        )
    )
    generator_score = [float(e) for e in generator_score]
    assert 1 <= min(generator_score) and max(generator_score) <= 5

    assert len(generator_ctx) == len(generator_hyp) == len(generator_ref) == len(generator_score)
    output_data = []

    for i in range(len(score)):
        c, h, r, s = [e.strip() for e in ctx[i].split("|||")], hyp[i], ref[i], float(score[i])
        assert 1 <= s <= 5
        s = (s * 10 - 10) / 40
        assert 0 <= s <= 1
        output_data.append(EvaluationExample(c, r, h, s, "ranker", None))
    for i in range(len(generator_score)):
        c, h, r, s = (
            [e.strip() for e in generator_ctx[i].split("|||")],
            generator_hyp[i],
            generator_ref[i],
            float(generator_score[i]),
        )
        assert 1 <= s <= 5
        s = (s * 10 - 10) / 40
        assert 0 <= s <= 1
        output_data.append(EvaluationExample(c, r, h, s, "generator", None))

    output_data = [asdict(e) for e in output_data]
    _write_items_to_jsonl(output_data, output_fname)
    _write_items_to_jsonl(output_data, output_fname.replace("/test_", "/valid_"))


def preprocess_convai2(dirname, output_fname, skip_ranker=False, skip_generator=False):
    """Preprocessing Convai2 dataset"""
    ranker_ctx_fname = os.path.join(dirname, "transformer_ranker/human_ctx.txt")
    ranker_hyp_fname = os.path.join(dirname, "transformer_ranker/human_hyp.txt")
    ranker_ref_fname = os.path.join(dirname, "transformer_ranker/human_ref.txt")
    ranker_score = os.path.join(dirname, "../human_score/convai2/transformer_ranker/human_score.txt")
    generator_ctx_fname = os.path.join(dirname, "transformer_generator/human_ctx.txt")
    generator_hyp_fname = os.path.join(dirname, "transformer_generator/human_hyp.txt")
    generator_ref_fname = os.path.join(dirname, "transformer_generator/human_ref.txt")
    generator_score = os.path.join(dirname, "../human_score/convai2/transformer_generator/human_score.txt")
    bert_ranker_ctx_fname = os.path.join(dirname, "bert_ranker/human_ctx.txt")
    bert_ranker_hyp_fname = os.path.join(dirname, "bert_ranker/human_hyp.txt")
    bert_ranker_ref_fname = os.path.join(dirname, "bert_ranker/human_ref.txt")
    bert_ranker_score = os.path.join(dirname, "../human_score/convai2/bert_ranker/human_score.txt")
    dialogGPT_ctx_fname = os.path.join(dirname, "dialogGPT/human_ctx.txt")
    dialogGPT_hyp_fname = os.path.join(dirname, "dialogGPT/human_hyp.txt")
    dialogGPT_ref_fname = os.path.join(dirname, "dialogGPT/human_ref.txt")
    dialogGPT_score = os.path.join(dirname, "../human_score/convai2/dialogGPT/human_score.txt")

    ctx, hyp, ref, score = list(
        map(
            read_txt,
            [ranker_ctx_fname, ranker_hyp_fname, ranker_ref_fname, ranker_score],
        )
    )
    assert len(ctx) == len(hyp) == len(ref) == len(score)
    generator_ctx, generator_hyp, generator_ref, generator_score = list(
        map(
            read_txt,
            [
                generator_ctx_fname,
                generator_hyp_fname,
                generator_ref_fname,
                generator_score,
            ],
        )
    )
    assert len(generator_ctx) == len(generator_hyp) == len(generator_ref) == len(generator_score)

    bert_ranker_ctx, bert_ranker_hyp, bert_ranker_ref, bert_ranker_score = list(
        map(
            read_txt,
            [bert_ranker_ctx_fname, bert_ranker_hyp_fname, bert_ranker_ref_fname, bert_ranker_score],
        )
    )
    assert len(bert_ranker_ctx) == len(bert_ranker_hyp) == len(bert_ranker_ref) == len(bert_ranker_score)

    dialogGPT_ctx, dialogGPT_hyp, dialogGPT_ref, dialogGPT_score = list(
        map(
            read_txt,
            [dialogGPT_ctx_fname, dialogGPT_hyp_fname, dialogGPT_ref_fname, dialogGPT_score],
        )
    )
    assert len(dialogGPT_ctx) == len(dialogGPT_hyp) == len(dialogGPT_ref) == len(dialogGPT_score)

    output_data = []

    for i in range(len(score)):
        c, h, r, s = [e.strip() for e in ctx[i].split("|||")], hyp[i], ref[i], float(score[i])
        assert 1 <= s <= 5
        s = (s * 10 - 10) / 40
        assert 0 <= s <= 1
        output_data.append(EvaluationExample(c, r, h, s, "ranker", None))
    for i in range(len(generator_score)):
        c, h, r, s = (
            [e.strip() for e in generator_ctx[i].split("|||")],
            generator_hyp[i],
            generator_ref[i],
            float(generator_score[i]),
        )
        assert 1 <= s <= 5
        s = (s * 10 - 10) / 40
        assert 0 <= s <= 1
        output_data.append(EvaluationExample(c, r, h, s, "generator", None))

    for i in range(len(bert_ranker_score)):
        c, h, r, s = (
            [e.strip() for e in bert_ranker_ctx[i].split("|||")],
            bert_ranker_hyp[i],
            bert_ranker_ref[i],
            float(bert_ranker_score[i]),
        )
        assert 1 <= s <= 5
        s = (s * 10 - 10) / 40
        assert 0 <= s <= 1
        output_data.append(EvaluationExample(c, r, h, s, "bert_ranker", None))

    for i in range(len(dialogGPT_score)):
        c, h, r, s = (
            [e.strip() for e in dialogGPT_ctx[i].split("|||")],
            dialogGPT_hyp[i],
            dialogGPT_ref[i],
            float(dialogGPT_score[i]),
        )
        assert 1 <= s <= 5
        s = (s * 10 - 10) / 40
        assert 0 <= s <= 1
        output_data.append(EvaluationExample(c, r, h, s, "dialogGPT", None))

    output_data = [asdict(e) for e in output_data]
    _write_items_to_jsonl(output_data, output_fname)
    _write_items_to_jsonl(output_data, output_fname.replace("/test_", "/valid_"))


if __name__ == "__main__":
    original_grade_path = "./data/evaluation/grade_eval_data/dailydialog/"
    os.makedirs("./data/evaluation/grade_eval_dd", exist_ok=True)
    preprocess_grade_dailydialogue(original_grade_path, "./data/evaluation/grade_eval_dd/test_processed.json")

    original_grade_path = "./data/evaluation/grade_eval_data/convai2/"
    os.makedirs("./data/evaluation/grade_eval_convai2", exist_ok=True)
    preprocess_convai2(original_grade_path, "./data/evaluation/grade_eval_convai2/test_processed.json")
