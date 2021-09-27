
## R code for validation to reproduce results in Table 4(b) of Section 5.2
## with pre-trained DNNs

## Change working directory to yours
setwd("~/R_code_DNN_hypothesis_testing_JCGS/") 
## Source functions with "DNN_stats_functions.r" saved in the same working directory
source("DNN_stats_functions.r")
## Load R depending R packages. Refer to "https://keras.rstudio.com/" for the installation
## of "keras".
library(doParallel)
library(keras)
library(reticulate)
library(tensorflow)
library(keras)
library(tibble)
library(car)

###############################################################################
set.seed(1) ## random seeds
## parameters
alpha = 0.05 ## nominal one sided type I error
n.sample.train = 85 ## first stage sample size per group
n.2.prop.cutoff = 0.1 # \theta_{min} cutoff in the adaptation rule
n.2.min = 28 # lower bound on n_2^{(min)}
n.2.max = 340 # n_2^{(max)}
n.test.H0.outer.itt = 10^6 # number of iterations of validating type I error and power

####################################################################################
## first DNN on test statistics
first.DNN.model = load_model_hdf5("MUSEC/MUSEC_first_DNN", 
                                  custom_objects = NULL, compile = TRUE)
## get the DNN weights
opt.model.weight = get_weights(first.DNN.model)
## get the number of layers 
DNN.first.opt.layers = length(opt.model.weight)/2-1

####################################################################################
## second DNN on critical values
second.DNN.model = load_model_hdf5("MUSEC/MUSEC_second_DNN", 
                                   custom_objects = NULL, compile = TRUE)
## get the DNN weights
opt.model.cutoff.weight = get_weights(second.DNN.model)
## get the number of layers 
DNN.cutoff.opt.layers = length(opt.model.cutoff.weight)/2-1

###################################################################################
## raed scale parameters
load(file = "MUSEC/MUSEC_scale_parameters")

col_means_train = col_mean_sd_two$col_means_train
col_stddevs_train = col_mean_sd_two$col_stddevs_train
col_means_cutoff_train = col_mean_sd_two$col_means_cutoff_train
col_stddevs_cutoff_train = col_mean_sd_two$col_stddevs_cutoff_train

####################################################################################
## validation of type I error rates and power
prop.val.vec = c(0.17, 0.27, 0.37, 0.47, rep(0.27, 3))
delta.prop.val.vec = c(rep(0, 4), 0.9, 1, 1.1)

val.para.grid = data.frame("prop_pbo" = prop.val.vec,
                           "delta_perc" = delta.prop.val.vec)

val.para.grid$ASN = 
val.para.grid$comb_power = val.para.grid$DNN_power = 
  val.para.grid$prop_trt = NA
n.val.ind = dim(val.para.grid)[1]

## a grid seach method on the critical value of ET method
critical.value.vec = seq(0.03, alpha, by = 0.001)
val.para.grid = cbind(val.para.grid, matrix(NA, nrow = n.val.ind,
                                            ncol = length(critical.value.vec)))
colnames(val.para.grid)[(7):dim(val.para.grid)[2]] = 
  paste0("naive_power_", critical.value.vec)

## evaluate each scenario
for (val.ind in 1:n.val.ind){
  print(val.ind)
  prop.val = val.para.grid$prop_pbo[val.ind] ## \theta_1 in group 1
  delta.perc.val = val.para.grid$delta_perc[val.ind] ## proportion of group difference

  ## base treatment effect as in the training stage
  delta.val.base = qnorm(alpha, 
       sd = sqrt(2*prop.val*(1-prop.val)/n.sample.train/2), 
       lower.tail = FALSE)-
    qnorm(0.85, sd = sqrt(2*prop.val*(1-prop.val)/n.sample.train/2),
          lower.tail = FALSE)
  
  ## treatment effect for validation
  prop.trt.val = min(prop.val + delta.val.base*delta.perc.val)
  val.para.grid$prop_trt[val.ind] = prop.trt.val
  
  ## generate validation data
  validation.mat = t(sapply(1:n.test.H0.outer.itt, function(x){
    ## simulate data for the first DNN
    data.val.fit = get.data.case.func(prop.grp.1.train.in = prop.val, 
                                      prop.grp.2.train.in = prop.trt.val, 
                                      n.in = n.sample.train, 
                                      if.test = TRUE)
      
    ## return the data and p-values for other methods
    val.return.vec = c(data.val.fit$data,
                           data.val.fit$test)
    return(val.return.vec)
  }))
  
  ## write data to a data frame
  validation.frame = data.frame(validation.mat[, c(1:5)])
  colnames(validation.frame) = c("grp1_mean_1", "grp2_mean_1",
                                 "grp1_mean_2", "grp2_mean_2",
                                 "n_2")

  ## write p-values from other methods to a data frame
  validation.test.frame = data.frame(validation.mat[, c(6:7)])
  colnames(validation.test.frame) = c("comb_test", "navie_test")
  
  ## get test statistics from the first DNN
  stats.val.vec = pred.DNN.normal(opt.model.weight, DNN.first.opt.layers, 
                             validation.frame,
                             col_means_train, col_stddevs_train)
  
  ## get \theta{12} under null
  prop.null = apply(validation.frame[, c("grp1_mean_1", "grp2_mean_1")], 1, mean)
  cutoff.val.data.input = data.frame("prop" = prop.null)
  
  ## calculate the corresponding critival value from the second DNN
  cutoff.val.vec = pred.DNN.normal(opt.model.cutoff.weight, DNN.cutoff.opt.layers, 
                                   cutoff.val.data.input,
                                  col_means_cutoff_train, 
                                  col_stddevs_cutoff_train)
  
  ## compute type I error rates / power for each method and ASN (per group).
  val.para.grid[val.ind, c("DNN_power")] = mean(stats.val.vec>=cutoff.val.vec)
  
  val.para.grid[val.ind, c("comb_power")] = 
    mean(validation.test.frame$comb_test<=alpha)
  
  val.para.grid[val.ind, c("ASN")] = 
    round(mean(validation.frame$n_2) + n.sample.train)
  
  ## compute decision rule of ET with each candidate critical values
  for (naive.ind in 1:length(critical.value.vec)){
    critival.value.temp = critical.value.vec[naive.ind]
    val.para.grid[val.ind, (paste0("naive_power_", critival.value.temp))] = 
      mean(validation.test.frame$navie_test<=critival.value.temp)
  }
  
  print(val.para.grid)
  write.csv(val.para.grid, 
            paste0("MUSEC/case_MUSEC_n_", n.sample.train, ".csv"))
}

write.csv(val.para.grid, 
          paste0("MUSEC/case_MUSEC_n_", n.sample.train, ".csv"))















