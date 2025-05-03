# GraphATC: advancing multilevel and multi-label anatomical therapeutic chemical classification via atom-level graph learning

[Wengyu Zhang](https://wengyuzhang.com), [Qi Tian](https://scholar.google.com/scholar?q=author:%22Qi%20Tian%22), [Yi Cao](https://academic.oup.com/bib/search-results?f_Authors=Yi+Cao), [Wenqi Fan](https://www.polyu.edu.hk/comp/people/academic-staff/prof-fan-wenqi/), [Dongmei Jiang](https://scholar.google.com/citations?user=Awsue7sAAAAJ), [Yaowei Wang](https://scholar.google.com/citations?user=o_DllmIAAAAJ), [Qing Li](https://www4.comp.polyu.edu.hk/~csqli/) and [Xiao-Yong Wei](https://www4.comp.polyu.edu.hk/~x1wei/).


[![Static Badge](https://img.shields.io/badge/DOI-10.1093%2Fbib%2Fbbaf194-blue)](https://doi.org/10.1093/bib/bbaf194)
[![Static Badge](https://img.shields.io/badge/Briefings%20in%20Bioinformatics-Volume%2026%2C%20Issue%202%2C%20March%202025-blue)](https://doi.org/10.1093/bib/bbaf194)
[![Static Badge](https://img.shields.io/badge/OUP-Open%20Access-green)](https://doi.org/10.1093/bib/bbaf194)
[![Static Badge](https://img.shields.io/badge/GitHub-GraphATC-blue)](https://github.com/lookwei/GraphATC)


[**Paper PDF**](https://academic.oup.com/bib/article-pdf/26/2/bbaf194/63012495/bbaf194.pdf) | [**Paper Website**](https://doi.org/10.1093/bib/bbaf194)

Official implementation of 'GraphATC: advancing multilevel and multi-label anatomical therapeutic chemical classification via atom-level graph learning', published in 'Briefings in Bioinformatics, Volume 26, Issue 2, March 2025' on 26 April 2025, [https://doi.org/10.1093/bib/bbaf194](https://doi.org/10.1093/bib/bbaf194)
<p align="center"><img width="850" src="images/GraphATC_pipeline.png"></p>

The accurate categorization of compounds within the anatomical therapeutic chemical (ATC) system is fundamental for drug development and fundamental research. Although this area has garnered significant research focus for over a decade, the majority
of prior studies have concentrated solely on the Level 1 labels defined by the World Health Organization (WHO), neglecting the labels of the remaining four levels. This narrow focus fails to address the true nature of the task as a multilevel, multi-label classification challenge. Moreover, existing benchmarks like Chen-2012 and ATC-SMILES have become outdated, lacking the incorporation of new drugs or updated properties of existing ones that have emerged in recent years and have been integrated into the WHO ATC system. To tackle these shortcomings, we present a comprehensive approach, GraphATC.

**Our contributions**:
- We have constructed the most extensive ATC dataset to date.
- We implement the multilevel, multi-label study by extending the task to Level-2 (i.e. L2).
- We build more accurate representations for polymers.
- We optimize the representation learning for macromolecular drugs.
- We build a more effective framework for aggregating component representations of multicomponent drugs.

**Table of contents**:
- [**Installation**](#installation) | [**Dataset**](#dataset) | [**Training**](#training) | [**Evaluation**](#evaluation)

## 📢 News

- **[2025.5.01]** The code and dataset of GraphATC has been released.
- **[2025.4.26]** Our paper has been published in Briefings in Bioinformatics.
- **[2025.4.07]** Our paper has been accepted by Briefings in Bioinformatics.

<a name="installation"></a>
## ⚙️ Installation
1. Clone the repository from GitHub.

```shell
git clone https://github.com/lookwei/GraphATC.git
cd GraphATC
```

2. Create conda environment.

```shell
conda create -n graphatc python=3.10
conda activate graphatc
```

3. Install packages.
```shell
conda install pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam/label/cu118 dgl

pip install -r requirements.txt
```

<a name="dataset"></a>

## 🗂️ Dataset

- Our constructed dataset `ATC-GRAPH` is available in [graphatc/dataset](./graphatc/dataset/) directory.
- To load the dataset, please refer to the [graphatc/dataset/uni_dataset.py](./graphatc/dataset/uni_dataset.py) file.

Comparison of ATC Benchmark Datasets:
|          | Dataset  | Chen-2012 [1] | ATC-SMILES [2] | **ATC-GRAPH (Ours)** |
| -------- | -------- | --------- | ---------- | ------------- |
| Group by | Year     | 2012      | 2022       | **2024**      |
| Polymer  | Non-Poly | 3852      | 4545       | **5259**      |
|          | Polymer  | 23        | 0          | **52**        |
| Mass     | Small    | 3715      | 4353       | **4822**      |
|          | Macro    | 160       | 192        | **489**       |
| #Comp    | Single   | 2275      | 2685       | **2931**      |
|          | Multiple | 1600      | 1860       | **2380**      |
|          | Total    | 3883      | 4545       | **5311**      |
|          | Coverage | 67.84%    | 79.40%     | **92.78%**    |


<a name="training"></a>

## 🚀 Training

1. GraphATC on ATC-GRAPH Level 1

```bash
bash scripts/train/train_GraphATC_L1.sh  
```

2. GraphATC on ATC-GRAPH Level 2

```bash
bash scripts/train/train_GraphATC_L2.sh  
```

The training log will be saved in the `graphatc/log/` directory.


<a name="evaluation"></a>

## 🏆 Evaluation

1. GraphATC on ATC-GRAPH Level 1

```bash
bash scripts/eval/eval_GraphATC_L1.sh  
```

2. GraphATC on ATC-GRAPH Level 2

```bash
bash scripts/eval/eval_GraphATC_L2.sh  
```

The evaluation results will be saved in the `graphatc/log/` directory.


## 📋 TODO
- [x] Create GraphATC repository;
- [x] Add brief introduction of the GraphATC;
- [x] Release the dataset;
- [x] Release the source code;
- [ ] Release the web server;
- [ ] Release more details;


## 📖 Citation
If you find the repository or the paper useful, please use the following entry for citation.

Wengyu Zhang, Qi Tian, Yi Cao, Wenqi Fan, Dongmei Jiang, Yaowei Wang, Qing Li, Xiao-Yong Wei, GraphATC: advancing multilevel and multi-label anatomical therapeutic chemical classification via atom-level graph learning, *Briefings in Bioinformatics*, Volume 26, Issue 2, March 2025, bbaf194, https://doi.org/10.1093/bib/bbaf194

```
@article{zhang2025graphatc,
  title={GraphATC: advancing multilevel and multi-label anatomical therapeutic chemical classification via atom-level graph learning},
  author={Zhang, Wengyu and Tian, Qi and Cao, Yi and Fan, Wenqi and Jiang, Dongmei and Wang, Yaowei and Li, Qing and Wei, Xiao-Yong},
  journal={Briefings in Bioinformatics},
  volume={26},
  number={2},
  pages={bbaf194},
  year={2025},
  publisher={Oxford University Press}
}
```

## References

[1] Chen L, Zeng WM, Cai YD, Feng KY, Chou KC. Predicting Anatomical Therapeutic Chemical (ATC) classification of drugs by integrating chemical-chemical interactions and similarities. *PLoS One*. 2012;7(4):e35254.

[2] Yi Cao, Zhen-Qun Yang, Xu-Lu Zhang, Wenqi Fan, Yaowei Wang, Jiajun Shen, Dong-Qing Wei, Qing Li, and Xiao-Yong Wei. Identifying The Kind Behind SMILES – Anatomical Therapeutic Chemical Classification using Structure-Only Representations,  *Briefings in Bioinformatics*, 2022, DOI:10.1093/bib/bbac346.
