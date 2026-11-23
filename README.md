🧬 RNA 3D Structure Prediction with RibonanzaNet 

Fine-tuning RibonanzaNet to predict the 3D coordinates of RNA molecules directly from their nucleotide sequence, built for the Stanford RNA 3D Folding (Part 2) Kaggle competition.

Python PyTorch Platform Task

📑 Table of Contents Overview Problem Statement Approach at a Glance Repository Structure Dataset Methodology Results Getting Started Configuration Reference Limitations and Future Work Acknowledgements Author 🔭 Overview 

RNA molecules fold into intricate 3D shapes that determine their biological function, from gene regulation to catalysis. Determining these shapes experimentally is slow and expensive, so accurate computational prediction would be a major step for drug discovery and RNA-based therapeutics.

This project takes a pretrained RNA foundation model, RibonanzaNet, and adapts it for 3D coordinate regression: given an RNA sequence of A/C/G/U nucleotides, the model outputs an (x, y, z) position for every residue.

The work is split into two notebooks:

Notebook Purpose rna-3d-folding-ribonanzanet-training.ipynb Data preparation, model definition, structure-aware loss functions and the fine-tuning loop rna-3d-folding-ribonanzanet-inference.ipynb Loads the fine-tuned weights, predicts structures for the test set (with sliding-window support for long RNAs) and writes submission.csv 🎯 Problem Statement 

Input: an RNA sequence (string over A, C, G, U).
Output: 3D coordinates (x, y, z) for each residue, with up to 5 candidate structures per target in the competition submission format.

Challenges addressed:

Variable length: sequences range from short fragments to very long RNAs. Missing coordinates: experimental structures contain gaps (NaNs) that must be handled in both filtering and loss computation. Rotation and translation invariance: a structure and its rotated copy are the same molecule, so the loss must not penalize global orientation. Memory limits: pairwise attention scales quadratically with length, so long sequences need cropping in training and windowing at inference. ⚡ Approach at a Glance flowchart LR A[RNA sequence<br/>A C G U] --> B[Tokenise<br/>A=0 C=1 G=2 U=3] B --> C[RibonanzaNet encoder<br/>9 ConvTransformer layers<br/>pairwise config] C --> D[256-d residue embeddings] D --> E[Linear 256 → 3<br/>xyz predictor] E --> F[3D coordinates<br/>per residue] F --> G[Sliding-window merge<br/>for long sequences] G --> H[submission.csv<br/>5 structure slots] Transfer learning: start from pretrained RibonanzaNet weights. New regression head: a linear layer maps the 256-dimensional residue embeddings to 3D coordinates. Invariant loss: combine a distance-matrix error (dRMAE) with an SVD-aligned coordinate error. Long-sequence handling: random crops during training, overlapping sliding windows during inference. 📁 Repository Structure . ├── rna-3d-folding-ribonanzanet-training.ipynb # Fine-tuning pipeline ├── rna-3d-folding-ribonanzanet-inference.ipynb # Prediction and submission ├── submission.csv # Generated predictions (after running inference) └── README.md 📊 Dataset 

Data comes from the Stanford RNA 3D Folding 2 competition on Kaggle (stanford-rna-3d-folding-2).

File Description train_sequences.csv / train_labels.csv Training sequences and per-residue 3D coordinates validation_sequences.csv / validation_labels.csv Held-out validation structures test_sequences.csv Sequences to predict sample_submission.csv Submission format template Data volumes observed in the notebooks Split Count Training structures processed 5,716 Validation structures processed 28 Test sequences 28 Total test residues (rows in submission) 9,762 Additional Kaggle inputs required Kaggle input Contents ribonanzanet2d-final Network.py (model class) and configs/pairwise.yaml ribonanzanet-weights Pretrained RibonanzaNet.pt ribonanzanet-finetuned Fine-tuned RibonanzaNet-3D.pt used at inference time 🧠 Methodology 1. Data preparation Labels are grouped by structure ID (pdb_id), and the first reference coordinates (x_1, y_1, z_1) per residue are used as regression targets. Structures are filtered to keep data quality high: Filter Rule Missing coordinates NaN ratio ≤ 10% Minimum length more than 10 residues Maximum length no upper limit in the filter (9,999,999); cropping handles length instead Nucleotides are tokenised as A→0, C→1, G→2, U→3. Sequences longer than max_len = 384 are randomly cropped to a 384-residue window each time they are loaded, acting as a mild form of data augmentation. 2. Model architecture Component Detail Backbone RibonanzaNet with the pairwise.yaml configuration Encoder depth 9 ConvTransformerEncoderLayer blocks Residue embedding size 256 Regression head nn.Linear(256, 3) Initialisation Pretrained RibonanzaNet weights Output One (x, y, z) per residue 3. Loss functions 

Predicting coordinates directly would be penalised for simply rotating or shifting a correct structure. The training objective therefore combines two complementary terms:

Loss What it measures Why it helps dRMAE (distance-matrix MAE) Mean absolute error between predicted and true pairwise distance matrices (scaled by Z=10) Fully invariant to rotation and translation; captures global fold Aligned SVD MAE MAE of coordinates after optimal rigid (Kabsch/SVD) alignment to ground truth Pushes absolute coordinates to match, with orientation removed loss = dRMAE(pred, gt) + align_svd_mae(pred, gt) 

NaN positions in the ground truth are masked out of both terms, and the diagonal of the distance matrix is excluded. The notebook also defines dRMSD and local_dRMSD (distance-clamped at 30 Å) as additional diagnostic and alternative objectives.

4. Training configuration Hyperparameter Value Optimiser AdamW (weight decay 0.0) Learning rate 3e-4 Scheduler Cosine annealing, starting after cos_epoch Gradient clipping max norm 1.0 Batch size 1 (variable-length structures) Max sequence length 384 Epochs (run configuration) 20 total, cosine decay over the last 5 (cos_epoch=15) Epochs (function default) 50 total, cos_epoch=35 Validation metric dRMAE on the validation set Checkpointing Best-validation weights saved, final weights saved at the end Seed 42, with deterministic cuDNN for reproducibility 5. Inference 

For each test sequence the model predicts coordinates in a single pass if the sequence fits in 384 residues. Longer sequences use a sliding window:

Step Detail Window size 384 residues Stride 192 (50% overlap) Tail handling An extra window anchored at the end of the sequence covers any remainder Merging Overlapping predictions are averaged per residue 

The final predictions are written to submission.csv in the competition format: ID, resname, resid, then x_k, y_k, z_k for k = 1…5.

📈 Results Run summary (from the notebooks) Item Value Training structures after parsing 5,716 Validation structures 28 Test sequences predicted 28 Submission shape 9,762 rows × 18 columns Submission columns 3 identifier columns + 5 × (x, y, z) = 18 Model RibonanzaNet, 9 ConvTransformer layers Inference status All 28 sequences processed successfully Sample prediction output 

First residues of target 8ZNQ, showing the predicted backbone trace (slot 1 shown):

ID resname resid x_1 y_1 z_1 8ZNQ_1 A 1 0.672 16.071 20.171 8ZNQ_2 C 2 2.558 11.287 19.241 8ZNQ_3 C 3 1.933 6.633 16.606 8ZNQ_4 G 4 -0.238 2.790 12.874 8ZNQ_5 U 5 -3.124 1.369 8.008 

Consecutive residues sit roughly 5 Å apart along the chain, consistent with the spacing expected between neighbouring nucleotides.

Performance metrics 

Fill in your own numbers from the training logs and the Kaggle leaderboard.

Metric Value Best validation dRMAE your value Final training loss (dRMAE + aligned MAE) your value Kaggle public leaderboard score (TM-score) your value Kaggle private leaderboard score (TM-score) your value Leaderboard rank your value Training curves 

Add your plots here, for example:

![Training curve](assets/training_curve.png)

🚀 Getting Started Option A: Run on Kaggle (recommended) Create a Kaggle notebook with a GPU accelerator. Add the competition data (stanford-rna-3d-folding-2) plus the three inputs listed under Additional Kaggle inputs. Training: open the training notebook, uncomment the __main__ block in Section 8 and run all cells. Save the best checkpoint as RibonanzaNet-3D.pt. Inference: attach the fine-tuned weights as a Kaggle dataset, run the inference notebook, and submit the generated submission.csv. Option B: Run locally git clone https://github.com/<your-username>/<your-repo>.git cd <your-repo> pip install torch numpy pandas pyyaml tqdm matplotlib plotly jupyter jupyter notebook 

Then update the paths in the Config class of each notebook to point at your local copies of the data, Network.py, pairwise.yaml and the weight files. A CUDA-capable GPU is required, since the notebooks call .cuda().

Starting a training run best_loss, best_predictions = train_model( model=model, train_dl=train_loader, val_dl=val_loader, epochs=20, cos_epoch=15, lr=3e-4, clip=1, ) print(f"Best Validation Loss: {best_loss:.4f}") ⚙️ Configuration Reference Parameter Default Description max_len 384 Crop length in training and window size at inference batch_size 1 Structures per step min_len_filter 10 Discard structures at or below this length max_len_filter 9,999,999 Effectively no upper length cap seed 42 Random seed num_tta 5 Number of test-time candidates configured at inference model_config_path .../configs/pairwise.yaml RibonanzaNet architecture config pretrained_weights_path .../RibonanzaNet.pt Starting weights for fine-tuning finetuned_weights_path .../RibonanzaNet-3D.pt Fine-tuned weights for inference 🔧 Limitations and Future Work 

Current limitations

Single deterministic prediction: the same structure is currently written into all five submission slots, so the five-candidate allowance is not exploited. One coordinate per residue: the head predicts a single point per nucleotide rather than a full atomic model. Windowed context for long RNAs: sliding windows only see local context, so very long molecules may lose global fold information. Sequence-only input: no multiple sequence alignment, secondary-structure or template information is used. 

Ideas for improvement

[ ] Generate diverse candidates (stochastic dropout, multiple checkpoints, test-time augmentation) to fill all five slots. [ ] Add template-based modelling for targets with close homologues in the training set. [ ] Incorporate secondary-structure predictions as auxiliary features or losses. [ ] Add a TM-score based validation metric that mirrors the competition scoring. [ ] Train on longer crops or use hierarchical stitching to preserve long-range contacts. [ ] Ensemble several fine-tuned checkpoints. 🙏 Acknowledgements Stanford RNA 3D Folding competition organisers and the Kaggle community. The RibonanzaNet authors for the pretrained RNA foundation model and open code. Kaggle for GPU compute. 👤 Author 

Chikonzero Biomedical Engineering graduate, Malawi University of Science and Technology (MUST) Focus: applied machine learning and medical AI

GitHub: @your-username Kaggle: your-kaggle-handle LinkedIn: your-profile 

⭐ If you found this project useful, consider giving the repository a star.



