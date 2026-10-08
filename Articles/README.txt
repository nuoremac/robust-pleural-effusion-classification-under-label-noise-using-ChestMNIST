PAPERS FOR THE PLEURAL-EFFUSION CLASSIFICATION PROJECT
Download/access check: 8 October 2026.

NEW PDF FILES
1. Sensoy_2018_Evidential_Deep_Learning.pdf
   EDL foundation, general classification. Supports our Dirichlet loss and
   uncertainty explanation, not a medical pleural-effusion benchmark.
2. Chen_2023_BoMD_Noisy_Chest_Xray.pdf
   ICCV 2023 method (arXiv first posted in 2022). Multilabel noisy chest X-ray
   classification; includes effusion-specific results. This is the arXiv copy.
3. Balaram_2022_CSEAL_Radiograph_Classification.pdf
   Earlier CSEAL paper: evidential active/semi-supervised classification on
   NIH-14. Do not cite this file as the extended 2026 journal paper.
4. Chest_Xray_Low_Resolution_Uncertain_Labels_2025_Preprint.pdf
   Agarwal and Sinha: low-resolution CheXpert CNN classification, including
   pleural effusion, and handling uncertain labels. The arXiv posting is 2025;
   a final publication venue/date has not been verified.

ADDITIONAL FULL TEXT
Widodo_SVM_Pleural_Effusion_FullText.xml contains the original full article
from Europe PMC, in JATS XML. It is not a PDF. The paper directly studies
SVM identification of pleural effusion. The original PDF links returned access
pages rather than PDFs, so no fake PDF was saved.
Article: https://doi.org/10.1016/j.heliyon.2023.e22778

PDFS NOT AVAILABLE THROUGH THE CHECKED LINKS
SNEL (2024): https://doi.org/10.1109/TMI.2024.3357986
CSEAL extension (2026): https://doi.org/10.1109/JBHI.2026.3696868
  Author preprint: https://doi.org/10.36227/techrxiv.175022082.23076736/v1
LRC-CXR (2026): https://doi.org/10.1016/j.eswa.2026.131438
Publisher/author links did not provide accessible PDFs; LRC-CXR has no open
PDF in the checked metadata. Access through a library or an author copy may
be necessary. No restricted-access PDF has been bypassed.

COMPARISON
paper_comparison.csv lists the task, approach, effusion relevance and intended
comparison for each paper. download_manifest.json records download outcomes.
For four currently readable approaches, use the pleural-effusion SVM (XML),
BoMD, earlier CSEAL and the low-resolution CNN study, with Sensoy as the EDL
foundation reference. If four PDF references are required, the four new PDF
files are available, but Sensoy is a general-method reference.

Separate published results from our measured results. Record class-specific
effusion ROC-AUC separately from mean multilabel ROC-AUC. Dataset, resolution,
label type, split, noise rate and annotation quality differ from ChestMNIST.
Do not infer algorithm superiority by ranking incompatible published scores.
Google Scholar citation counts and a most-cited ranking are not verified.
Preserve the distinction between recent arXiv posting and publication year.

The three pre-existing clinical-etiology papers remain unchanged.
