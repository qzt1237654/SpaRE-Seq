from setuptools import setup, find_packages

setup(
    name="spare_seq",               
    version="1.0.0",                
    author="qzt1237654",            
    author_email="2861658363@qq.com", 
    description="A comprehensive bioinformatics pipeline for Spatial RNA Editing Analysis.",
    
    packages=find_packages(),       
    
    install_requires=[              
        # Base Data Science
        "pandas",
        "numpy",
        "scipy",
        "tqdm",
        "scikit-learn",  # Needed for PCA in clustering
        
        # Bioinformatics & Single Cell
        "gffutils",      # Needed for Genomic Annotation
        "scanpy",        # Needed for Clustering and UMAP
        "anndata",       # Needed for reading .h5ad files
        
        # Visualization
        "matplotlib",
        "seaborn",       # Needed for beautiful boxplots and scatter plots
        "adjustText",    # Needed for automatic text labeling in Volcano plots
        
        # R Integration & Deep Learning
        "rpy2",          # Needed for running DESeq2 via R
        "torch",         # Needed for spCLUE model training
    ],
    
    python_requires=">=3.8",        
)