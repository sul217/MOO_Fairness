# Fair Binary Classification using Stochastic Multi-Objective Gradient Descent Algorithm

This is the numerical implementation for the work about **Accuracy and Fairness Trade-offs in Machine Learning**. Refer to the [paper](https://arxiv.org/pdf/2008.01132.pdf) for more details. 

## 1. Package requirements

The code was implemented using Python 3.6
- numpy
- math
- time
- matplotlib
- random
- scipy
- quadprog (required for more than two objectives problem)


## 2. Jupyter notebooks

We provide three jupyter notebooks corresponding to four sets of trade-off results. The corresponding multi-objective formulations are presented inside the notebook markdown. 

`Synthetic_data.ipynb` 
Disparate impact vs prediction loss, output the trade-off results for a set of randomly generated data.

`Adult_data.ipynb `  
Disparate impact vs prediction loss, output trade-off results for single binary-valued sensitive attribute--gender and multi-valued categorical sensitive attribute--Race for Adult income dataset. Note that the multi-valued part takes longer time (around one hour) to get the Pareto front. Also, the trade-off results are shown in terms of testing data. 

`COMPAS_data.ipynb`  
Equal opportunity vs prediction loss, output trade-off results for single binary-valued sensitive attribute for COMPAS two-year dataset. Training and testing dataset coincides due to the shortage of data instance. 

For each notebook, one can run all the cells with the giving parameters to get the trade-offs. The pre-run results are also contained in these notebooks. The results might differ from each run due to the stochasticity.


## 3. Python scripts

The .py files serve as data preprocessing, PF-SMG algorithm, and bi-objective function/gradient formulations and computation.

- PFSMG.py     detailed implementations of PF-SMG algorithm, explanation on parameters.
- functions.py    bi-objective formulation including objective functions (logistic loss and fairness approximation), gradients, etc. for different settings
- generate_synthetic_data.py  imported by Synthetic_data.ipynb for generating synthetic data.
- data/load_compas_data.py  details on the COMPAS data processing, only for reference.


## 4. Data 

Data files are attached in the folder called 'data'.

- `testData_seed6600451_param5.05.txt`  
synthetic data randomly generated using generate_synthetic_data.py

- `Adult_income_gender_reduced.txt`, `Adult_income_race_reduced_smg.txt`, and `Adult_income_race&gender_reduced_smg.txt`  
Original dataset is downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult), preprocessed using one-hot ecoding and normalization, remove sensitive features "gender", "race", and a highly predictive feature--"fnlwgt", reduce dimension of "education" by combining severals categorical values and "country" into two classes--US and non-US.

- `CAMPAS_race.txt`  
The origin data can be found at [the github page](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv), see `load_compas_data.py` file for data preprocessing details ([Credit to Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, and Krishna P. Gummadi](https://github.com/mbilalzafar/fair-classification/tree/master/disparate_mistreatment/propublica_compas_data_demo).

## 5. Examples
The figure below shows the full trade-off between accuracy and fairness w.r.t. disparate impact using Adult income dataset and taking gender as the sensitive attribute. <img src="https://latex.codecogs.com/svg.latex?\large&space;f_1(x)" title="\large f_1(x)"/> and <img src="https://latex.codecogs.com/svg.latex?\large&space;f_2(x)" title="\large f_2(x)"/> refer to prediction loss and squared convariance approximation for disparate impact. 

<img src="data/Adult_gender_DI.png" width="600px" style="float: right;">


## In case you cite our work please refer to the paper:

S. Liu and L. N. Vicente.  Accuracy and fairness trade-offs in machine learning:  A stochastic multi-objective approach.ISE Technical Report 20T-016, Lehigh University, 2020.
