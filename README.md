# EEGETGoldilocksDataComposition

This is the Github Repo for the data processing step for the research done in Too Fine or Too Coarse? The Goldilocks Composition of Data Complexity for Robust Left-Right Eye-Tracking Classifiers by Brian Xiang and Abdelrahman Abdelmonsef @ **Swarthmore College**. 

## Downloading the datasets

1. Navigate to https://osf.io/ktv7m/. 
2. On the left, you will find two folders, the first of which is named "Dropbox: EEGEyeNet." If this first folder has a (+) sign next to it, click on the (+) sign. Otherwise, go to the next step directly. 
3. Four folders should pop up. Open the folder named **Prepared** (Click the + sign next to the folder named "prepared"). 
4. Scroll until you see four files that start with "LR_task_with."
5. Download the second to last one to your device ("LR_task_with_antisaccade_synchronised_min.npz").
6. Scroll until you see eight files that start with "Direction_task_with."
5. Download the third one to your device ("Direction_task_with_dots_synchronised_min.npz").
7. Place the downloaded datasets into the folder named **Tasks**.

## Python Requirements

Verify that your python environment contains proper installations for the three files listed below:
1. general_requirments.txt
2. standard_ml_requirments.txt
3. tensorflow_requirements.txt

## Data Processing

1. Run **TranslatingLR.py**. You should now have a file called 'LR_task_with_dots_synchronised_min.npz' in the **Tasks** folder.
2. The remaining code is built upon the coding interface provided by the EEGEyeNet repository: https://github.com/ardkastrati/EEGEyeNet. A detailed explanation of the interface is provided there. 
3. Adjust the **config.py** file to select training/testing set composition, specifically the fields 'PA_train_ratio' and 'PA_test_ratio.' NOTE: DO NOT CHANGE ANY OTHER PARAMETERS, our code is not meant to be able to test any other parameters.
5. Once the training and testing datasets have been determined, run **main.py** and wait for the results in the folder called 'runs'.

If you have any comments or suggestions, contact bxiang1@swarthmore.edu and/or ayahia1@swarthmore.edu.


 
