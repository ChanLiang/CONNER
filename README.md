# Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators
Welcome to the repository for our EMNLP 2023 paper, "Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators." In this work, we introduce **CONNER** (COmpreheNsive kNowledge Evaluation fRamework), a systematic approach designed to evaluate the output of Large Language Models (LLMs) across key dimensions such as Factuality, Relevance, Coherence, Informativeness, Helpfulness, and Validity.

Here, you'll find the necessary code and resources to replicate our findings and further explore the potential of LLMs. We hope they help facilitate your work in exploring the frontiers of LLMs with a touch of ease.

## CONNER Framework


### Intrinsic Evaluation

- **Factuality:** Assessing the verifiability of the information against external evidence.
- **Relevance:** Ensuring the knowledge aligns with the user's query intent.
- **Coherence:** Evaluating the logical flow of information at both sentence and paragraph levels.
- **Informativeness:** Measuring the novelty or unexpectedness of the knowledge provided.

### Extrinsic Evaluation

- **Helpfulness:** Gauging whether the knowledge aids in enhancing performance on downstream tasks.
- **Validity:** Certifying the factual accuracy of downstream task results when utilizing the knowledge.

## Getting Started

### Setting Up the Environment

Begin by setting up your Conda environment with the provided `environment.yaml` file, which will install all necessary packages and dependencies.

```bash
conda env create -f env/environment.yaml -n CONNER
conda activate CONNER
```
If you run into any missing packages or dependencies, please install them as needed.

#### Evaluating Your LLMs
Run the evaluation script that corresponds to your dataset and chosen metric. Replace ${data} with your dataset choice (nq or wow) and ${metric} with one of the following metrics: factuality, relevance, info, coh_sent, coh_para, validity, helpfulness.
```bash
# Run evaluation script. Example usage:
# bash scripts/nq_factuality.sh
# bash scripts/wow_relevance.sh
bash scripts/${data}_${metric}.sh
```
#### Viewing Results
```bash
# Display the evaluation results. Example usage:
# bash scripts/nq_factuality_view.sh
# bash scripts/wow_relevance_view.sh
bash scripts/${data}_${metric}_view.sh
```

## Citing Our Work
If you find our work helpful in your research, please citing our paper:
```
@misc{chen2023factuality,
      title={Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators}, 
      author={Liang Chen and Yang Deng and Yatao Bian and Zeyu Qin and Bingzhe Wu and Tat-Seng Chua and Kam-Fai Wong},
      year={2023},
      eprint={2310.07289},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
