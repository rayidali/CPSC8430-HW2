# CPSC8430-HW2
# Video Caption Generation using Seq2Seq Models

This project focuses on generating descriptive captions for video content, leveraging the power of Sequence-to-Sequence (Seq2Seq) models with Attention Mechanisms. Developed as part of the CPSC 8430: Deep Learning course, this work aims to bridge video content with natural language descriptions.

## Requirements

To run this project, you will need:

- Python 3.6+
- PyTorch 1.7+
- NumPy
- CUDA (for GPU acceleration)

## Dataset

The dataset comprises 1450 training videos and 100 test videos, alongside corresponding JSON files containing labels for training and evaluation. This data is essential for training the Seq2Seq model to generate accurate and relevant video captions.

## Model Architecture

The model consists of two main components:
- **Encoder**: Encodes the video features into a context-rich representation.
- **Decoder**: Generates a caption from the encoded video features, using an Attention Mechanism to focus on relevant parts of the video at each step of the generation process.



