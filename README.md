# Six Approaches to Improve BERT for Claim Verification as Applied to the Fact Extraction and Verification Challenge (FEVER) Dataset

The paper for this project is [here](https://jonathan-ling.github.io/artefacts/2021.03%20BERT%20FEVER.pdf).

## Abstract
BERT, a transformer-based model often used in natural language processing, has been used in various research for fact extraction and verification tasks, but suffers from various issues when applied to claim verification. In this project, we aimed to implement the BERT model for claim verification on the FEVER (Fact Extraction and Verification) dataset, and suggest and implement six improvement approaches to the original BERT model - pre-processing evidence via data augmentation (synonym replacement and back-translation), changing the transformer settings (BERT vs DistilBERT and number of epochs), and post- processing its results neurally. While our modifications did not result in significant changes to the FEVER score, likely due to BERT's strong pre-training, applying our neural aggregation layer improved performance on the DistilBERT model, a lighter version of the BERT model.

## Baseline code
|     File   / repository                              |     GitHub Link                                                                                                   |     Associated paper / documentation                                                                                                            |
|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
|     Baseline   BERT                                  |     [Link](https://github.com/simonepri/fever-transformers)                                                               |     [FEVER Transformers](https://github.com/simonepri/fever-transformers)                                                                                                     |
|     Synonym   replacement                            |     [Link](https://github.com/jasonwei20/eda_nlp/blob/5d54d4369fa8db40b2cae7d490186c057d8697f8/experiments/nlp_aug.py)    |     [EDA: Easy Data Augmentation Techniques for   Boosting Performance on Text Classification Tasks](https://arxiv.org/pdf/1901.11196.pdf)    |
|     Aggregator   for claim verification labelling    |     [Link](https://github.com/takuma-ynd/fever-uclmr-system/blob/interactive/neural_aggregator.py)                        |     [Four Factor Framework For Fact Finding (HexaF)](https://www.aclweb.org/anthology/W18-5515.pdf)                                          |

## Changes to baseline code

|     Experiments                                       |     File                                             |     Summary of changes made                                                                                             |
|-------------------------------------------------------|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
|     Synonym   replacement     &   Back-translation    |     src/pipeline/claim-verification/generate.py      |     Added synonym replacement and back-translation code                                                                 |
|     Aggregator                                        |     src/pipeline/claim-verification/model.py         |     Added prediction scores for each class (refutes,   supports, not enough information) for each retrieved sentence    |
|     Aggregator                                        |     src/pipeline/claim-verification/aggregator.py    |     Created neural network model to aggregate claims to   replace the original if-else model                            |
|     Common   to all                                   |     scripts/pipeline.sh                              |     Set up experiments to be able to be run at the   command line with appropriate flags                                |

## Scripts to run

Instead of running ```bash scripts/pipeline.sh claim_verification --model-type bert --model-name bert-base-cased``` in the original instructions ([Primarosa, 2020](https://github.com/simonepri/fever-transformers)), run the following commands for the appropriate experiment:

### Synonym replacement
```
bash scripts/pipeline.sh replace_synonyms
bash scripts/pipeline.sh claim_verification   --model-type bert --model-name bert-base-cased
```

### Back-translation
```
bash scripts/pipeline.sh backtranslation
bash scripts/pipeline.sh claim_verification --model-type   bert --model-name bert-base-cased
```

### Aggregation layer
```
bash scripts/pipeline.sh claim_verification   --model-type bert --model-name bert-base-cased
bash scripts/pipeline.sh write_predictions   --model-type bert --model-name bert-base-cased
bash scripts/pipeline.sh aggregator
```