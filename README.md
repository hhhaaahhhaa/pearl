## PEARL: Optimize Prompt Selection for Enhanced Answer Performance Using Reinforcement Learning
### Reinforcement Learning 2023 Final Project
#### Group5

* For preparing the environment.

``` 
pip install -r requirements.txt
```

* For preparing the data, download the preprocessed dataset, create and put into _data folder.


* The original dataset is in:
``` 

https://huggingface.co/datasets/kuanhuggingface/hint-lm-data


```

* The preprocessed dataset is in:

``` 

https://drive.google.com/drive/folders/1qwgZ5w62DwF9JVHu7SGD-PjqCkAobrhy


```

* Run gen.py to construct the desired preprocessed datasets or you can directly use the preprocessed dataset listed above.


``` python

python gen.py

```

* For training, you can modify some parameters directly in the code.
``` python

python rl_train.py

```

* For evaluation, you can modify some parameters directly in the code.
``` python

python new_eval.py

```

* For ploting the evaluation results, you can modify some parameters directly in the code.

``` python

python analysis/scatter_plot.py

```

* For ploting the baseline results, you can modify some parameters directly in the code.

``` python

python analysis/tradeoff.py

```