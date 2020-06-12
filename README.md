# american-prison-writing_nlp

## Table of Contents
- [Overview](#overview)
- [Project](#project)
- [Acknowledgements](#acknowledgements)

## Overview:
America has 5 percent of the world’s population and 25 percent of the world’s incarcerated population. By the latest figures, the US holds approximately 2.2 million people in prisons and jails. This number doesn’t capture the full impact of the criminal justice system – another 11 million people are part of the carceral system on probation or parole, not to mention the families and communities that have been ravaged and disrupted by mass incarceration. 

Despite the astonishing scale of the problem, very little is known or understood about the experience of currently or formerly incarcerated people. Those not directly impacted have limited understanding of the realities of incarceration - poor living conditions, inhumane treatment, violence, abuse by guards, wrongful convictions, or discriminatory practices, to name a few.

Several groups have taken the lead on combating this ignorance. One such initiative is the [American Prison Writing Archive (APWA)](https://apw.dhinitiative.org/) – an open-source database for currently and formerly incarcerated people and prison staff to document their experiences. These essays give the authors a voice and give the readers an insight into the lives of people who are easily ignored, marginalized, or forgotten by society.

For my Computational Content Analysis final paper, I conduct a content analysis of the APWA. Using counting, classification, structured topic modeling, and word embedding techniques, this paper attempts a systematic exploration of this important and unique corpus. None of these techniques can act as substitutes for thoroughly reading and understanding the often-difficult material contained in the essays – doing so would be a disservice to an already-marginalized population. However, content analysis techniques can be used as a useful first step in exploring and generating hypotheses about the essays and what themes they contain.

## Project:
- data_ingest.py: contains functions to scrape the APWA data
- helper_functions.py: contain functions used throughout the analysis
- counting_divergence.ipynb: notebook conducting word counting and divergence analysis on the corpus
- classification.ipynb: notebook attempting to use classification methods to predict author demographics
- structured_modeling.Rmd and structured_modeling.html: R markdown and output for structured modeling analysis
- word_embedding.ipynb: training word2vec and doc2vec models to look beyond demographic categories

## Acknowledgements
I want to thank and acknowledge the course staff of SOCI 40133 (Professor James Evans, Hyunku Kwon, and Bhargav Desikan) for their support and feedback. I also want to thank the American Prison Writing Archive for the fantastic work they are doing in collecting and transcribing these essays. 
