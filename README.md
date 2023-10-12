## Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators
This repository contains code for evaluating LLMs-generated knowledge using CONNER.

## CONNER Framework

![My Image](figs/framework.png)

### Intrinsic Evaluation

- *_Factuality:_* Whether the information in the knowledge can be verified by external evidence.
- *_Relevance:_* Whether the knowledge is relevant to the user query.
- *_Coherence:_* Whether the knowledge is coherent at the sentence and paragraph levels.
- *_Informativeness:_* Whether the knowledge is new or unexpected against the model's existing knowledge.

### Extrinsic Evaluation

- *_Helpfulness:_* Whether the knowledge can improve the downstream tasks.
- *_Validity:_* Whether the results of downstream tasks using the knowledge are factually accurate.

### Usage
Please stay tuned, updates will be coming soon.

## Citation

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
