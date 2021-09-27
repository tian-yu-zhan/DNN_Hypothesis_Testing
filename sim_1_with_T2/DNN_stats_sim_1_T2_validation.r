
## R code for validation to reproduce results of "DNN-T2" in the first column 
## of Table 1 in Section 4.1 with pre-trained DNNs

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
library(kernlab)

###############################################################################
## set random seeds
set.seed(1)
## parameters
alpha = 0.05 ## nominal one-sided type I error
n.sample.train = 20 ## sample size
n.test.H0.outer.itt = 10^6 # number of iterations of validating type I error and power
## The results of MMD with 10^4 iterations have been generated in "sim_1" folder. 
## We set simulation iterations for MMD at 10 to save computational time. 
n.test.MMD.outer.itt = 10^1 

###################################################################################
## first DNN on test statistics
first.DNN.model = load_model_hdf5("sim_1_with_T2/sim_1_first_DNN", 
                                  custom_objects = NULL, compile = TRUE)
## get the DNN weights
opt.model.weight = get_weights(first.DNN.model)
## get the number of layers 
DNN.first.opt.layers = length(opt.model.weight)/2-1

####################################################################################
## second DNN on critical values
second.DNN.model = load_model_hdf5("sim_1_with_T2/sim_1_second_DNN", 
                                  custom_objects = NULL, compile = TRUE)
## get the DNN weights
opt.model.cutoff.weight = get_weights(second.DNN.model)
## get the number of layers 
DNN.cutoff.opt.layers = length(opt.model.cutoff.weight)/2-1

###################################################################################
## raed scale parameters
load(file = "sim_1_with_T2/sim_1_scale_parameters")

col_means_train = col_mean_sd_two$col_means_train
col_stddevs_train = col_mean_sd_two$col_stddevs_train
col_means_cutoff_train = col_mean_sd_two$col_means_cutoff_train
col_stddevs_cutoff_train = col_mean_sd_two$col_stddevs_cutoff_train

####################################################################################
## validation on type I error rates and power
theta.val.vec = c(1, 5) ## \theta_1 in group 1
## varying proportions on the traing group difference
delta.prop.val.vec = c(0, 0.8, 1, 1.1) 
k.val.vec = c(0.2, 0.8) ## known design parameters k
val.para.grid = data.frame(expand.grid(delta.prop.val.vec, k.val.vec,
                                       theta.val.vec
                                       ))
val.para.grid = val.para.grid[, 3:1]
colnames(val.para.grid) = c("theta", "k", "delta.prop")
## select design features as in Table 1
# val.para.grid = val.para.grid[(val.para.grid$theta==1&val.para.grid$k==0.2)|
#                                 (val.para.grid$theta==5&val.para.grid$k==0.8)  , ]

val.para.grid$DNN_validation_hour = val.para.grid$T2_power = 
  val.para.grid$LRT_power = val.para.grid$MMD_power = 
val.para.grid$wil_power = val.para.grid$t_power = val.para.grid$DNN_power = 
  val.para.grid$theta_trt = NA
n.val.ind = dim(val.para.grid)[1]

## validations on each scenario
for (val.ind in 1:n.val.ind){
  print(val.ind)
  theta.val = val.para.grid$theta[val.ind] ## \theta_1 in group 1
  delta.prop.val = val.para.grid$delta.prop[val.ind] ## proportion of difference
  k.val = val.para.grid$k[val.ind] ## known design parameter
  
  ## obtain base treatment effect in the training stage
  sd.val.base = (2*k.val*theta.val)/sqrt(12) ## sd of uniform distribution
  
  delta.val.base = qnorm(alpha, 
                         sd = sqrt(2*sd.val.base^2/n.sample.train), lower.tail = FALSE)-
    qnorm(0.45, sd = sqrt(2*sd.val.base^2/n.sample.train),lower.tail = FALSE)
  
  ## treatment effect for validation with varying proportion of group difference
  theta.val.trt = theta.val + delta.val.base*delta.prop.val
  val.para.grid$theta_trt[val.ind] = theta.val.trt
  
  ## generate validation data
  validation.mat = t(sapply(1:n.test.H0.outer.itt, function(x){
    ## simulate data for the first DNN
    data.val.fit = get.data.unif.T2.func(theta.grp.1.in = theta.val, 
                                            theta.grp.2.in = theta.val.trt, 
                                            k.in = k.val,
                                            n.in = n.sample.train,
                                            if.test = TRUE,
                                            if.MMD = FALSE
             )
    
    ## return data vector and p-values from other two tests
    val.return.vec = c(data.val.fit$data,
                       data.val.fit$mean,
                           data.val.fit$test)
    return(val.return.vec)
  }))
  
  ## write data to a data frame
  validation.frame = data.frame(validation.mat[, c(1:6)])
  # colnames(validation.frame) = c(unlist(lapply(1:2, 
  #      function(x){paste0(c("grp1_", "grp2_")[x], 
  #                         c("min","sd"))})), "k", "T2")
  
  ## MMD
  MMD.dec.vec = t(sapply(1:n.test.MMD.outer.itt, function(x){
    ## simulate data for the first DNN
    data.val.fit = get.data.unif.T2.func(theta.grp.1.in = theta.val, 
                                      theta.grp.2.in = theta.val.trt, 
                                      k.in = k.val,
                                      n.in = n.sample.train,
                                      if.test = FALSE,
                                      if.MMD = TRUE
    )
    
    ## return data vector and p-values from other two tests
    val.return.vec = c(data.val.fit$test[3])
    return(val.return.vec)
  }))
  
  ## write p-values from two methods to a data frame
  validation.test.frame = data.frame(validation.mat[, c(8:9, 11:12)])
  colnames(validation.test.frame) = c("t.test", "wilcox.test", "LRT.stats","T2.stats")
  
  time.val.start = Sys.time()
  ## get test statistics from the first DNN
  stats.val.vec = pred.DNN.normal(opt.model.weight, DNN.first.opt.layers, 
                             validation.frame,
                             col_means_train, col_stddevs_train)
  
  ## get pooled \theta_{12} under H_0
  theta.null = as.numeric(validation.mat[, 7])
  cutoff.val.data.input = data.frame("mean" = theta.null, "k" = k.val)
  
  ## calculate the corresponding critival value from the second DNN
  cutoff.val.vec = pred.DNN.normal(opt.model.cutoff.weight, DNN.cutoff.opt.layers, 
                                   cutoff.val.data.input,
                                  col_means_cutoff_train, 
                                  col_stddevs_cutoff_train)
  
  time.val = difftime(Sys.time(), time.val.start, 
                      units = c("hours"))
  
  ## write type I error rates / power to the final output table
  val.para.grid[val.ind, c("DNN_power")] = mean(stats.val.vec>=cutoff.val.vec)
  val.para.grid[val.ind, c("t_power")] = mean(validation.test.frame$t.test<=alpha)
  val.para.grid[val.ind, c("wil_power")] = mean(validation.test.frame$wilcox.test<=alpha)
  val.para.grid[val.ind, c("MMD_power")] = mean(MMD.dec.vec)
  
  ## critical values for LRT and T2
  if (k.val ==0.2) {LRT.cutoff = 1.037; T2.cutoff = 1.0306}
  if (k.val ==0.8) {LRT.cutoff = 1.1067; T2.cutoff = 1.1056}
  print(mean(validation.test.frame$LRT.stats>1.106))
  print(mean(validation.test.frame$T2.stats>1.1056))
  
  val.para.grid[val.ind, c("LRT_power")] = 
    mean(validation.test.frame$LRT.stats>LRT.cutoff)
  val.para.grid[val.ind, c("T2_power")] = 
    mean(validation.test.frame$T2.stats>T2.cutoff)
  val.para.grid[val.ind, c("DNN_validation_hour")] = time.val
  
  print(val.para.grid)
  write.csv(val.para.grid, 
            paste0("sim_1_with_T2/sim_1_n_", n.sample.train, "_with_T2.csv"))
}

write.csv(val.para.grid, 
          paste0("sim_1_with_T2/sim_1_n_", n.sample.train, "_with_T2.csv"))














