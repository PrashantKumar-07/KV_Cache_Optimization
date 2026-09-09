# longbench_metrics.py — scoring functions for LongBench benchmark
#
# Extracted from THUDM/LongBench evaluation code (used identically by
# SnapKV, Quest, and H2O). Kept as standalone copy so we don't depend
# on SOTA repo paths at runtime.

import re
import string
from collections import Counter
from rouge import Rouge


def normalize_answer(s):
    """Normalize: lowercase, strip articles/punctuation/whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction, ground_truth, **kwargs):
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_score(prediction, ground_truth, **kwargs):
    """ROUGE-L F1 via the real `rouge` package, on RAW (unnormalized) text --
    matches the reference LongBench eval (sota/snapkv/.../metrics.py:104-110)
    exactly. The previous version ran a hand-written LCS-based ROUGE-L over
    normalize_answer()'d text (lowercased, punctuation/article-stripped),
    which is a QA-style normalization, not the summarization-ROUGE
    convention the cited protocol uses -- numbers from that version are not
    directly comparable to published LongBench/SnapKV gov_report/samsum
    results.
    """
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def classification_score(prediction, ground_truth, **kwargs):
    """Matches the reference (sota/snapkv/.../metrics.py:89-102): find every
    class name that's a substring of the (untruncated) prediction, drop any
    match that's itself a substring of another candidate ground truth in the
    match list, then split credit 1/n across whatever ambiguity remains --
    not a binary first-match exact test.
    """
    em_match_list = []
    all_classes = kwargs.get("all_classes", [])
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    """Matches the reference (sota/snapkv/.../metrics.py:56-66): fractional
    credit = (# numbers in prediction equal to the ground-truth paragraph
    id) / (total numbers extracted from prediction) -- penalizes a
    prediction that lists many candidate numbers, unlike a binary
    "somewhere in the text" check.
    """
    match = re.search(r"Paragraph (\d+)", ground_truth)
    if not match:
        return 0.0
    ground_truth_id = match.group(1)
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for n in numbers if n == ground_truth_id)
    return right_num / len(numbers)


def count_score(prediction, ground_truth, **kwargs):
    """Matches the reference (sota/snapkv/.../metrics.py:47-54): fractional
    credit = (# numbers in prediction equal to ground truth) / (total
    numbers extracted), not binary credit for the first number matching.
    """
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for n in numbers if n == str(ground_truth).strip())
    return right_num / len(numbers)


# Map task names to scoring functions
DATASET_TO_METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
}

# Max generation length per task (from LongBench)
DATASET_TO_MAXGEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32,
}

# Prompt templates per task (from LongBench)
DATASET_TO_PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the following question based on the above story, as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\nAnswer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing the constraints and conditions for the desired summary. Write a summary of the meeting transcript that satisfies the query.\n\nTranscript:\n{context}\n\nQuery: {input}\nSummary:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news.\n\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the following dialogue.\n\n{context}\n\nSummary:",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many distinct paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, e.g., 1, 2, 3, and so on.\n\nThe count of unique paragraphs is:",
    "passage_retrieval_en": "The following are some paragraphs, each indicated by a numerical identifier []. Please read them and determine which paragraph the following summary corresponds to.\n\n{context}\n\nThe above is a set of paragraphs. Below is a summary.\n\nSummary: {input}\n\nPlease enter the identifier of the paragraph that the summary corresponds to. The answer format must be like \"Paragraph [id]\".",
}
