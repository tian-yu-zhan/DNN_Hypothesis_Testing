# DNN_Hypothesis_Testing

This is a help file for the R code accompanying a paper with the title “Finite-Sample Two-Group Composite Hypothesis Testing via Machine Learning”.

The R code of reproducing simulation results in the main article is saved at the folder “sim_1” for “DNN” method and “sim_1_with_T2” for “DNN-T2” method in Table 1 of Section 4.1, “sim_2_df” for “DNN” method and “sim_2_df_LRT” for “DNN-LRT” method in Table 2 of Section 4.2, “sim_3” for Table 3 of Section 4.3, “ACTT” for Table 4(a) of Section 5.1, and “MUSEC” for Table 4(b) of Section 5.2.

Training: Within a specific example folder, source “XX_training.r” to generate TS-DNN for statistics, CV-DNN for critical values, and a file for scaling parameters. Note that those three files are already saved in each example folder.

Validation: One can directly source “XX_validation.r” to generate validation results based on the pre-trained models.

