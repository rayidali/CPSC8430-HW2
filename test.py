import sys
import torch
import json
from eval import VideoCaptionDataset as LoadTestData, test as EvaluateModel, Seq2Seq as Seq2Seq, Encoder as Encoder, Decoder as Decoder, Attention as Attention
from torch.utils.data import DataLoader
from bleu_eval import BLEU as ComputeBLEU
import pickle

# Load the pre-trained model
trained_model_path = 'SavedModel/model0.h5'
pretrained_model = torch.load(trained_model_path, map_location=lambda storage, loc: storage)

# Setup for testing data
test_features_path = 'MLDS_hw2_1_data/testing_data/feat'
test_dataset = LoadTestData(f'{sys.argv[1]}')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

# Load index to word mapping
with open('i2w.pickle', 'rb') as mapping_file:
    idx_to_word = pickle.load(mapping_file)

# Prepare model for evaluation
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model = pretrained_model.to(compute_device)

# Generate captions
generated_captions = EvaluateModel(test_dataloader, pretrained_model, idx_to_word)

# Output generated captions to file
with open(sys.argv[2], 'w') as output_file:
    for video_id, caption in generated_captions:
        output_file.write(f'{video_id},{caption}\n')

# Load test labels for BLEU score computation
test_labels_path = '/scratch1/nsuresh/DL/testing_label.json'
test_labels = json.load(open(test_labels_path))
evaluation_output = sys.argv[2]
caption_results = {}

# Parse the generated captions from output file
with open(evaluation_output, 'r') as result_file:
    for line in result_file:
        line = line.strip()
        vid_id, generated_caption = line.split(',', 1)
        caption_results[vid_id] = generated_caption

# Compute BLEU scores for each video and calculate average
bleu_scores = []
for video_item in test_labels:
    individual_scores = []
    reference_captions = [caption.rstrip('.') for caption in video_item['caption']]
    individual_scores.append(ComputeBLEU(caption_results[video_item['id']], reference_captions, True))
    bleu_scores.append(individual_scores[0])

average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score: {average_bleu:.2f}")
