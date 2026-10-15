import json
import re
import sys
from pathlib import Path
from collections import Counter

INCORRECT = {
    "or_46_0_0", "or_240_0_6", "or_94_0_0",
    "or_140_0_1", "or_106_0_2",
}
UNCERTAIN = {
    "or_57_0_3", "or_244_0_5", "or_229_0_3",
    "or_65_0_0", "or_147_0_4",
}

def normalise(text):
    return re.sub(r"\s+", " ", text).strip()

def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

input_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(input_path.read_text(encoding="utf-8-sig"))
candidates = []
needs_review = []
seen = set()
reasons_count = Counter()

for article_i, article in enumerate(data["data"]):
    for passage_i, paragraph in enumerate(article["paragraphs"]):
        passage = paragraph["context"].strip()

        for question_i, qa in enumerate(paragraph["qas"]):
            review_id = f"or_{article_i}_{passage_i}_{question_i}"
            question = qa["question"].strip()
            answers = [
                answer.get("text", "").strip()
                for answer in qa.get("answers", [])
            ]
            answers = [answer for answer in answers if answer]

            row = {
                "review_id": review_id,
                "original_id": qa.get("id"),
                "article_id": article_i,
                "passage_id": f"or_{article_i}_{passage_i}",
                "category": qa.get("category"),
                "passage": passage,
                "question": question,
                "answers": answers,
            }

            reasons = []

            if not passage or not question:
                reasons.append("empty_passage_or_question")
            if not answers:
                reasons.append("blank_answer")
            elif not any(answer in passage for answer in answers):
                reasons.append("answer_not_found_in_passage")
            if review_id in INCORRECT:
                reasons.append("incorrect_in_review")
            if review_id in UNCERTAIN:
                reasons.append("uncertain_in_review")

            duplicate_key = (normalise(passage), normalise(question))
            if duplicate_key in seen:
                reasons.append("duplicate_passage_question")
            else:
                seen.add(duplicate_key)

            if reasons:
                row["review_reasons"] = reasons
                needs_review.append(row)
                reasons_count.update(reasons)
            else:
                candidates.append(row)

save_jsonl(output_dir / "candidate_qa.jsonl", candidates)
save_jsonl(output_dir / "needs_review.jsonl", needs_review)

print("Candidate QA pairs:", len(candidates))
print("Set aside for review:", len(needs_review))
print("Reasons:", dict(reasons_count))