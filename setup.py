from setuptools import setup, find_packages

setup(
    name="spare_seq",               
    version="0.1.0",                
    author="qzt1237654",            
    author_email="2861658363@qq.com", 
    description="A comprehensive bioinformatics pipeline for Spatial RNA Editing Analysis.",
    
   
    packages=find_packages(),       
    
    
    install_requires=[              
        "pandas",
        "numpy",
        "scipy",
        "tqdm",
        "gffutils",
        "matplotlib"
    ],
    
    python_requires=">=3.8",        
)