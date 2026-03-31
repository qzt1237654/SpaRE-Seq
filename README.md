# SpaRE-Seq: Spatial RNA Editing Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Under%20Development-orange)

**SpaRE-Seq** is a comprehensive bioinformatics pipeline designed to integrate spatial transcriptomics data (`.h5ad`) with A-to-I RNA editing matrices. It enables high-resolution spatial differential editing analysis, gene-editing correlation, and genomic annotation at the cluster level.

## 🌟 Key Features

The pipeline is structured into three main analytical workflows:

* **Workflow A: Stress-Induced Differential Editing**
    * Aggregates Bin-level A/G matrices into Cluster-level matrices.
    * Implements rigorous dual-filtering (Median A+G > 10, G signal present in ≥ 3 samples).
    * Utilizes **DESeq2** to identify significant Differential Editing Sites (DESS) between control and stress groups.
* **Workflow B: Gene-Editing Correlation Analysis**
    * Extracts spatial expression matrices of key editing enzymes (*Adar, Adarb1, Adarb2*).
    * Calculates **Spearman/Pearson** correlations between editing levels and enzyme expression.
    * Visualizes intersecting sites across samples using UpSet plots.
* **Workflow C: Spatial-Specific Editing Dynamics**
    * Performs One-vs-Rest (e.g., Cluster 0 vs. Clusters 1-8) matrix reconstruction.
    * Identifies spatially restricted editing sites under baseline/control conditions.
* **Genomic Annotation Module**
    * Maps significant sites to genomic regions (CDS, UTRs, Exons, Introns) using `GTF` databases.
    * Generates publication-ready stacked bar plots and volcano plots.


## SpaRE-Seq Core Environment Dependencies
torch==1.13.1
numpy==1.23.5
scanpy==1.9.3
anndata==0.8.0
pandas==1.5.3
scipy==1.10.0
scikit-learn==1.2.2
tqdm==4.64.1
matplotlib==3.7.0
seaborn==0.12.2
jupyter==1.0.0
gffutils==0.13
rpy2==3.4.1

## 🚀 Installation (Coming Soon)

*Note: SpaRE-Seq is currently undergoing refactoring into a standard Python package. Once completed, it will be installable via pip:*

```bash
# Future installation method
git clone [https://github.com/qzt1237654/SpaRE-Seq.git](https://github.com/qzt1237654/SpaRE-Seq.git)
cd SpaRE-Seq
pip install .

## 📜 Acknowledgments

This pipeline incorporates and builds upon excellent existing open-source tools and tutorials. Special thanks to:

* **spCLUE**: The spatial clustering and integration steps were inspired by and adapted from the [spCLUE integration tutorial](https://github.com/EnchantedJoy/spCLUE/blob/main/tutorial/integration.ipynb).
