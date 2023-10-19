# SciTechNews: A Dataset for Scientific Journalism

This repository provides the dataset and code to reproduce the work presented in the paper [‘Don’t Get Too Technical with Me’: A Discourse Structure-Based Framework for Science Journalism](link). [Accepted to EMNLP 2023]

The SciTechNews dataset consists of scientific papers paired with their corresponding
press release snippet mined from [ACM TechNews](https://technews.acm.org/).
ACM TechNews is a news aggregator that provides regular news digests about scientific achieve-
ments and technology in the areas of Computer Science, Engineering, Astrophysics, Biology, and
others.



## Download

1. HuggingFace Datasets

The tokenized
Available as [ronaldahmed/scitechnews](https://huggingface.co/datasets/ronaldahmed/scitechnews)

```
from datasets import load_dataset

dataset = load_dataset("ronaldahmed/scitechnews")

```


2. External links
- [Tokenized](https://drive.google.com/file/d/1Ve43Ur3NYqND1iAIb9L_KDMCVaW7v3OF/view?usp=sharing)
- [Untokenized](https://drive.google.com/file/d/1XK-KF5raBYzXnQMYfvTXEX3cqmi_IOkT/view?usp=sharing)

```
{
	"id": String,          			 # unique ID
	"pr-title": String,    		 	 # Title as found in the ACMTECHNEWS website
	"pr-article": String,  			 # Press release article
	"pr-summary": String,  		 	 # Press release summary
	"sc-title": String,    		 	 # Title of scientific article
	"sc-abstract": String, 		 	 # Abstract of scientific article
	"sc-sections": List[String], 	 # List of sections in the scientific article
	"sc-section_names": List[String] # List of section names
	"sc-authors": List[String] 		 # list of authors' name and affiliations, in the format '<name> | <affil>'
}
```

Paragraphs in the press release articles (`pr-article`) and sections of the scientific article (`sc-sections`)
are separated by `\n`.<br>

