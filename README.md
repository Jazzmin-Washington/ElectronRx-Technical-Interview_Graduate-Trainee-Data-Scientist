# ElectronRx-Technical-Interview_Graduate-Trainee-Data-Scientist
Our goal is to extract haemodynamic parameters from optical signals. This challenge is
related to photoplethysmography (PPG) waveforms related to cardiovascular functions.
Each PPG represents volumetric variations of blood circulation. Insights into variations
or deviation from the baseline of the overall pattern provides invaluable diagnostic and
therapeutic intervention opportunities.

_________________________________________________________________________________________________________________
## Day 1 - Research - Review of Literature
- The first step in the process is to research previous cadiac studies using photoplethysmography and the methods previous studies used for analysis, 

Summary of PPG: PPG values are generally transmitted from Wearable Patient Monitoring via mobile and web applications. PPG is advantageous as it eliminates many issues such as invasive procedures typical of continuous monitoring and uncomfortable non-continuous cuff-based readings. PPG uses human skin vessels to find changes in light transmitted or reflected through photoelectric sensors providing different aspects of cardiac survellienve including: 
- Blood Oxygen Saturation
- Heart Rate
- Stress Levels
- Cardiac Output
- Blood Pressure Estimation\
- Respiration
- Arterial Aging
- Microvascular Flow
- Factors contributing to Atherosclerosis and CVD
- Autonomic Function

The robustness of PPG readings circulation gives way to valuabler nsights into variations
or deviation from the baseline of the overall pattern provides cutting-eedge diagnostic and
therapeutic intervention opportunities (Sadad et al, 2022, and Georgieva-Tsaneva et al, 2022)

---------------------------------------------------------------------------------------------------------------

#### Preprocessing: 
- It is impertative the PPG signals are preprocessed. 
- To perform preprocessing, Digital Filtering must be done to remove noise and other perturbations.
  - Noise hinders the extraction of reliable information from PPG signaling.
  - Noise can be a result of individual characteristics(skin thickness, age,  etc.), physiological processes (respiration, location, temperature, etc.) or external disturbance within the environment.  The most common filtering technique for biomedical processing is a moving average filter.
  - It is also important to check the signal quality and remove any motion artefacts as this can skew the data
  
  
- Once preprocessing is done, a methodology to analyse the PPG signals needs to be selected. 
---------------------------------------------------------------------------------------------------------------
#### Methodology For the Analysis of Atrial Fibrillation
 
This project will focus on identifying atrial fibrillation. Based on previous studies, Tree-Based Algorithms, Support Vector Machines, and Neural Networks/Deep Learning would be the most beneficial in designing an Algorithm (Priyadarshini et al, 2021, Georgieva-Tsaneva et al, 2022, Huang et al, 2022). More specifically, when analysing whether pulse wave present atrial fibrillation there are three steps:
1. Feature extraction - one or more features from each pulse wave is needed to produce a time series of measurements. 
    - Interbeat Intervals and Pulse Amplitudes are commonly used. 
    - Heart Rate Variability - Measures Interbeat Intervals (IBI) over time but it is important to also remove any outliers and ectopic beats for all features. 
    - Outlier Removal Method: comparing consecutive IBIs and ensuring they have a normal-to-normal ratios.
 
2. Calculate Summary Statistics - Statistics such as mean value of the time series or variability such as standard deviation. 
    - Typically these can be linear or nonlinear
    - Poincare plot is most often used
  
3. Classification: Pulse waves are classified as sinus rhythm or atrial fibrillation. 
    - Traditional statistical methods can be used such as pre-specified thresholds or a logistic regression models
    - Other more computational methods can also be used such as Time Domain Analysis, Frequency Domain Analysis, Time-Frequency Analysis and Machine Learning. 
      - Previous studies provided great results with Support Vector Machines, Tree-Based Algorithms, CNN-LSTM, and other deep learning models. 
        - With Machine Learning , deep learning can be used to classify a segment of PPG signals directly as an input/feature to the model. Accelerometry data can be used in cases of small sample size. The model is then fine-tuned or transfer learning techniques can be used. 
 -------------------------------------------------------------------------------------------------------------
 #### Additional Reading to be Completed:
 - L. M. Eerikainen ¨ et al., “Atrial fibrillation monitoring with wrist-worn photoplethysmography-based wearables: State-of-the-art review,” Cardiovascular Digital Health Journal, vol. 1, no. 1, pp. 45–51, 2020. https://doi.org/10.1016/j.cvdhj.2020.03.001  
- S.-C. Tang et al., “Identification of atrial fibrillation by quantitative analyses of fingertip photoplethysmogram,” Scientific Reports, vol. 7, no. 1, p. 45644, 2017. https://doi.org/10.1038/srep45644 
- T. Pereira et al., “Photoplethysmography based atrial fibrillation detection: a review,” npj Digital Medicine, vol. 3, no. 1, p. 3, 2020. https://doi.org/10.1038/s41746-019-0207-9 
- P. H. Charlton, “Detecting atrial fibrillation from the photoplethysmogram,” 2021. https://commons.wikimedia.org/wiki/File:Detecting atrial fibrillation (AF) from the photoplethysmogram (PPG).svg  
- M. V. Perez et al., “Large-scale assessment of a smartwatch to identify atrial fibrillation,” New England Journal of Medicine, vol. 381, no. 20, pp. 1909–1917, 2019. https://doi.org/10.1056/NEJMoa1901183  
- Y. Guo et al., “Mobile photoplethysmographic technology to detect atrial fibrillation,” Journal of the American College of Cardiology, vol. 74, no. 19, pp. 2365–2375, 2019. https://doi.org/10.1016/j.jacc.2019.08.019  

  
      

