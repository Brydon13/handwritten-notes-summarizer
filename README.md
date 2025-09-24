# Handwritten Notes Summarizer

## Overview

This repository contains a tool for recognizing and summarizing handwritten notes, specifically tailored for students. The project aims to bridge the gap between handwritten note-taking preferences and digital convenience by providing a solution that converts handwritten notes into digital text and generates concise summaries, with emphasis on highlighted segments.

## Contributors

This was a group project for the ENSE412 Machine Learning course at the University of Regina. Group members include:

- Brydon Herauf
- Nathan Cameron
- Meklit Alemu

## Key Features

- **OCR (Optical Character Recognition):** Convert handwritten notes into digital text.
- **NLP (Natural Language Processing):** Generate summaries of handwritten notes, with emphasis on highlighted segments.
- **Highlight Segmentation:** Identify parts of the handwritten notes that are highlighted, aiding in the summarization process.

## Usage

This project was created with limited time and resources, and therefore has issues dealing with certain types of input. Feel free to clone and experiment with the model, but know that the results may be imperfect. You will need your own API keys for certain features. The process goes as follows:

1. Upload a scanned image of handwritten notes to the tool.
2. The OCR model will convert the handwritten text into digital format.
3. NLP will generate a concise summary of the notes, with emphasis on highlighted segments.
4. Review and edit the generated summary as needed.

Check out final-report.pdf for an example with our greatest results!

Please note that this repository contains a collection of our most important scripts. If you are interested in running it, you will need to fill in some missing pieces.

## Acknowledgements

- Special thanks to [OpenAI](https://openai.com) for their contributions to the field of NLP.
- Credits to [mltu by Python Lessons](https://github.com/pythonlessons/mltu) for the deep learning architecture and utilities used to train our model.
