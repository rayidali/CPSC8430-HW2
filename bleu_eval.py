import math
import operator
import sys
import json
from functools import reduce

def calculate_ngram_overlap(test_sentence, reference_sentences, ngram_size):
    overlap_count = 0
    total_ngrams = 0
    ref_length_total = 0
    test_length_total = 0
    for index in range(len(test_sentence)):
        ref_ngram_counts = []
        ref_sentence_lengths = []
        for ref in reference_sentences:
            ref_text = ref[index]
            ngram_counts = {}
            tokens = ref_text.strip().split()
            ref_sentence_lengths.append(len(tokens))
            for start in range(len(tokens) - ngram_size + 1):
                ngram = ' '.join(tokens[start:start + ngram_size]).lower()
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            ref_ngram_counts.append(ngram_counts)

        test_text = test_sentence[index]
        test_ngrams = {}
        tokens = test_text.strip().split()
        for start in range(len(tokens) - ngram_size + 1):
            ngram = ' '.join(tokens[start:start + ngram_size]).lower()
            test_ngrams[ngram] = test_ngrams.get(ngram, 0) + 1

        overlap_count += calculate_clip_count(test_ngrams, ref_ngram_counts)
        total_ngrams += len(tokens) - ngram_size + 1
        ref_length_total += match_closest_length(ref_sentence_lengths, len(tokens))
        test_length_total += len(tokens)

    precision = float(overlap_count) / total_ngrams if total_ngrams > 0 else 0
    brevity_pen = calculate_brevity_penalty(test_length_total, ref_length_total)
    return precision, brevity_pen

def calculate_clip_count(test_ngram_counts, reference_ngram_counts):
    clip_count = 0
    for ngram, count in test_ngram_counts.items():
        max_ref_count = max((ref_counts.get(ngram, 0) for ref_counts in reference_ngram_counts), default=0)
        clip_count += min(count, max_ref_count)
    return clip_count

def match_closest_length(reference_lengths, test_length):
    differences = [(abs(test_length - ref_length), ref_length) for ref_length in reference_lengths]
    return min(differences)[1]

def calculate_brevity_penalty(candidate_length, reference_length):
    if candidate_length > reference_length:
        return 1
    else:
        return math.exp(1 - (float(reference_length) / candidate_length))

def calculate_geometric_mean(precisions):
    return reduce(operator.mul, precisions) ** (1 / len(precisions))

def compute_BLEU(candidate, references, multiple_references=False):
    bleu_score = 0.
    candidate_sentences = [candidate.strip()]
    reference_sentences = [[ref.strip()] for ref in references] if multiple_references else [[references.strip()]]
    ngram_precision, brevity_penalty = calculate_ngram_overlap(candidate_sentences, reference_sentences, 1)
    bleu_score = calculate_geometric_mean([ngram_precision]) * brevity_penalty
    return bleu_score

if __name__ == "__main__":
    test_data = json.load(open('testing_label.json', 'r'))
    submission_file = sys.argv[1]
    evaluation_result = {}
    with open(submission_file, 'r') as file:
        for line in file:
            line = line.rstrip()
            comma_index = line.index(',')
            video_id = line[:comma_index]
            generated_caption = line[comma_index + 1:]
            evaluation_result[video_id] = generated_caption

    bleu_scores = []
    for item in test_data:
        video_scores = []
        reference_captions = [caption.rstrip('.') for caption in item['caption']]
        video_scores.append(compute_BLEU(evaluation_result[item['id']], reference_captions, True))
        bleu_scores.append(video_scores[0])

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print("Average BLEU score: " + str(average_bleu))
