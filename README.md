# CommonLit Readability Prize with AllenNLP

Kernel: https://www.kaggle.com/himako/allennlp-jsonnet-weak-example-using-roberta


### Training

```
poetry run allennlp train training_configs/model_v{1,2,3}.jsonnet -s serialization  # 1, 2, or 3
```
