# Google Colab Tutorial
Please follow the tutorial on using KnAC in text-clustering and fil in the survey at the end: [Colab Notebook](https://colab.research.google.com/drive/1SJaG_wW0h1_JaPk40vPNP3dpTJGa1xXG)
# Knowledge Augmented Clustering (KnAC)
KnAC is a toolk for expert knowledge extension with a usage of automatic clustering algorithms.
It allows to refine expert-based labeling with splits and merges recommendations of expert labeling augmented with explanations.
The explanations were formulated as rules and therefore can be easily interpreted incorporated with expert knowledge.

The overall workflow for KnAC is presented in Figure below:
![Workflow for KnAC](./pix/workflow.png?raw=true "Title")

## Set up environment to run examples
Some of the packages used in KnAC anre not available in conda, hence the following code should set up all of the requirements in virtual environment:
```
conda create --name knac python=3.8
conda activate knac
conda install pip
pip install -r requirements.txt
```
After that run `jupyter lab` and anvigate to `examples`   direcotry to run notebooks.
