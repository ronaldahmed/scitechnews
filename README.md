# SciTechNews: A Dataset for Scientific Journalism

This repository provides the dataset and code to reproduce the work presented in the paper [‘Don’t Get Too Technical with Me’: A Discourse Structure-Based Framework for Science Journalism](link). [Accepted to EMNLP 2023]

The SciTechNews dataset consists of scientific papers paired with their corresponding
press release snippet mined from [ACM TechNews](https://technews.acm.org/).
ACM TechNews is a news aggregator that provides regular news digests about scientific achieve-
ments and technology in the areas of Computer Science, Engineering, Astrophysics, Biology, and
others.


## Supported Tasks and Leaderboards

This dataset was curated for the task of Science Journalism, a text-to-text task where the input is a scientific article and the output is a press release summary.
However, this release also include additional information of the press release and of the scientific article, such as
press release article body, title, authors' names and affiliations.

The science juornalism leaderboard is found [here]().



## Dataset Structure


### Data Fields

```
{
	"id": String,          			 # unique ID
	"pr-title": String,    		 	 # Title as found in the ACMTECHNEWS website
	"pr-article": String,  			 # Press release article
	"pr-summary": String,  		 	 # Press release summary
	"sc-title": String,    		 	 # Title of scientific article
	"sc-abstract": String, 		 	 # Abstract of scientific article
	"sc-article": String,  			 # Concatenated abstract and sections of the scientific article
	"sc-sections": List[String], 	 # List of sections in the scientific article
	"sc-section_names": List[String] # List of section names
	"sc-authors": List[String] 		 # list of authors' name and affiliations, in the format '<name> | <affil>'
}
```


### Example Instance


```
{
  "id": 37,
  "pr-title": "What's in a Developer's Name?",
  "pr-article": "In one of the most memorable speeches from William Shakespeare's play, Romeo and Juliet , Juliet ponders, \" What's in a name? That which...",
  "pr-summary": ""Researchers at the University of Waterloo's Cheriton School of Computer Science in Canada found a software developer's perceived race and ethnicity,...",
  "sc-title": On the Relationship Between the Developer's Perceptible Race and Ethnicity and the Evaluation of Contributions in OSS",
  "sc-abstract": "Context: Open Source Software (OSS) projects are typically the result of collective efforts performed by developers with different backgrounds...",
  "sc-article": "Context: Open Source Software (OSS) projects are typically the result of .... In any line of work, diversity regarding race, gender, personality...",
  "sc-sections": ["In any line of work, diversity regarding race, gender, personality...","To what extent is the submitter's perceptible race and ethnicity related to...",...],
  "sc-section_names": ["INTRODUCTION", "RQ1:", "RQ2:", "RELATED WORK",...],
  "sc-authors": ["Reza Nadri | Cheriton School of Computer Science, University of Waterloo", "Gema Rodriguez Perez | Cheriton School of ...",...]
}
```


### Data Splits

Number of instances in train/valid/test are 26,368/1431/1000.<br>
Note that the training set has only press release data (`pr-*`), however
splits validation and test do have all fields.

### Data Statistics

In the table below, the statistics of our dataset are separated by *non-aligned* and *aligned*.
*Non-aligned* data means that the press release article could not be linked to a scientific article, whereas *aligned* means that a datum has both PR article/summary fields as well as scientific article information.

|                        | Instances | #doc-words | #summ-words | #summ-sents |
|------------------------|----------:|-----------:|------------:|------------:|
| PR non-aligned         |   29069    | 612.56  | 205.93  | 6.74 |
| PR aligned             | 2431 | 780.53  | 176.07  | 5.72 |
| Sci. aligned           | 2431 | 7570.27  | 216.77  | 7.88 |


## Download

1. HuggingFace Datasets

The untokenized dataset is available as [ronaldahmed/scitechnews](https://huggingface.co/datasets/ronaldahmed/scitechnews)

```
from datasets import load_dataset

dataset = load_dataset("ronaldahmed/scitechnews")

DatasetDict({
    train: Dataset({
        features: ['id', 'pr-title', 'pr-article', 'pr-summary', 'sc-title', 'sc-article', 'sc-abstract', 'sc-section_names', 'sc-sections', 'sc-authors'],
        num_rows: 26638
    })
    validation: Dataset({
        features: ['id', 'pr-title', 'pr-article', 'pr-summary', 'sc-title', 'sc-article', 'sc-abstract', 'sc-section_names', 'sc-sections', 'sc-authors'],
        num_rows: 1431
    })
    test: Dataset({
        features: ['id', 'pr-title', 'pr-article', 'pr-summary', 'sc-title', 'sc-article', 'sc-abstract', 'sc-section_names', 'sc-sections', 'sc-authors'],
        num_rows: 1000
    })
})

```

Paragraphs in the press release articles (`pr-article`) and sections of the scientific article (`sc-sections`)
are separated by `\n`. Data is not sentence or word tokenized.<br>
Note that field `sc-article` includes the article's abstract as well as its sections.

2. External links
- [Untokenized](https://drive.google.com/file/d/1vjbrQKQHUmDFO-pzzvwzA42FayV2a5PL/view?usp=sharing)
- [Tokenized](https://drive.google.com/file/d/1hILN6vWag9C9SPrDMWlqOgvd16KTWHt9/view?usp=sharing)

Tokenization and sentence splitting was done using [spaCy](https://spacy.io/).
Paragraphs in the press release articles (`pr-article`) and sections of the scientific article (`sc-sections`)
are separated by `\n\n`; sentences are separated by `\n`.
Similarly to the untokenized version, field `sc-article` includes the article's abstract as well as its sections.


## Dataset Creation

### Source Data

Press release snippets are mined from ACM TechNews and their respective scientific articles are mined from 
reputed open-access journals and conference proceddings.


### Initial Data Collection and Normalization

We collect archived TechNews snippets between 1999 and 2021 and link them with their respective press release articles. 
Then, we parse each news article for links to the scientific article it reports about.
We discard samples where we find more than one link to scientific articles in the press release.
Finally, the scientific articles are retrieved in PDF format and processed using [Grobid](https://github.com/kermitt2/grobid). 
Following collection strategies of previous scientific summarization datasets, section heading names are retrieved, and the article text is divided into sections. We also extract the title and all author names and affiliations.


## Considerations for Using the Data

### Social Impact of Dataset

The task of automatic science journalism is intended to support journalists or the researchers themselves in writing high-quality journalistic content more efficiently and coping with information overload.
For instance, a journalist could use the summaries generated by our systems as an initial draft and edit it for factual inconsistencies and add context if needed.
Although we do not foresee the negative societal impact of the task or the accompanying data itself, we point at the 
general challenges related to factuality and bias in machine-generated texts, and call the potential users and developers of science journalism 
applications to exert caution and follow up-to-date ethical policies.  


## Dataset Curators

- Ronald Cardenas, University of Edinburgh
- Bingsheng Yao, Rensselaer Polytechnic Institute
- Dakuo Wang, Northeastern University
- Yufang Hou, IBM Research Ireland


## Reproducing Experiments in Paper

### Preprocessed Data


The preprocessed data our main model, Bart_plan, is available [here](https://drive.google.com/file/d/14U5g6l-Ex1NKvVTktQ2r-IiNemsiftRc/view?usp=sharing), as a json line format.
Each datum contains two fields, `article` and `pr_summary`.<br>
Field `article` is composed of the concatenated abstract and introduction of the scientific article, prepended with author metadata, and with each sentence prepended with its scientific rhetorical label (e.g. background, conclusion).<br>
Field `pr_summary` contains the PR summary prepended by the oracle content plan.
Please see the example below.

<p align="middle">
    <img src="img/input-plan-example.png" alt="Example of enriched source and target" width="800"/>
</p>

### Training Models

To train Bart-Plan, please take a look at script `scripts/train_bart-plan.sh`.


### Inference

#### Generating content plan and summary
In order to run a pretrained model and generate the complete target (content plan + summary), run `scripts/predict_bart-plan.sh`.


#### Generating a summary with a custom plan
In order to generate a PR summary conditioned on a predetermined plan, run `gen_from_custom_plan.sh`.
The generator reads a data file with fields `article` (same format as the preprocessed data above) and `plan`, which has only the content plan.
See `data_custom_plan/valid-toy-plan.json` for a toy example.


#### Postprocessing
Running inference with Bart-plan generates a content plan immediately followed by a summary in the same string.
We provide a post-processing script to separate plan and summary easily. Simply pass the output json file to the script:

```
python post_process_plan_preds.py -p "<prediction-json-file>"

```


## Citation Information

If our dataset or models are useful to you, please cite the following:

Provide the [BibTex](http://www.bibtex.org/)-formatted reference for the dataset. For example:
```
@article{cardenas2023dont,
      title={'Don't Get Too Technical with Me': A Discourse Structure-Based Framework for Science Journalism}, 
      author={Ronald Cardenas and Bingsheng Yao and Dakuo Wang and Yufang Hou},
      year={2023},
      eprint={2310.15077},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


