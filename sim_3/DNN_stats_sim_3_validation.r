
## R code for validation to reproduce results in Table 3 of Section 4.3
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
## set random seeds
set.seed(1)
## parameters
alpha = 0.05 ## nominal one sided type I error
n.sample.train = 50 ## sample size
n.test.H0.outer.itt = 10^6 # number of iterations of validating type I error and power

###################################################################################
## first DNN on test statistics
first.DNN.model = load_model_hdf5("sim_3/sim_3_first_DNN", 
                                  custom_objects = NULL, compile = TRUE)
## get the DNN weights
opt.model.weight = get_weights(first.DNN.model)
## get the number of layers 
DNN.first.opt.layers = length(opt.model.weight)/2-1

####################################################################################
## second DNN on critical values
second.DNN.model = load_model_hdf5("sim_3/sim_3_second_DNN", 
                                   custom_objects = NULL, compile = TRUE)
## get the DNN weights
opt.model.cutoff.weight = get_weights(second.DNN.model)
## get the number of layers 
DNN.cutoff.opt.layers = length(opt.model.cutoff.weight)/2-1

###################################################################################
## raed scale parameters
load(file = "sim_3/sim_3_scale_parameters")

col_means_train = col_mean_sd_two$col_means_train
col_stddevs_train = col_mean_sd_two$col_stddevs_train
col_means_cutoff_train = col_mean_sd_two$col_means_cutoff_train
col_stddevs_cutoff_train = col_mean_sd_two$col_stddevs_cutoff_train


####################################################################################
## validation of type I error rates and power 
mu.val.vec = c(-0.5, 0)
sd.val.vec = c(1, 1.5)
delta.prop.val.vec = c(0, 0.8, 1, 1.2)
val.para.grid = data.frame(expand.grid(delta.prop.val.vec,
                                       sd.val.vec,
                                       mu.val.vec
                                       ))

val.para.grid = val.para.grid[, 3:1]
colnames(val.para.grid) = c("mu", "sd", "delta.prop")

val.para.grid$t_power = val.para.grid$DNN_power = 
  val.para.grid$mu_trt = NA
n.val.ind = dim(val.para.grid)[1]

## evaluate each scenario
for (val.ind in 1:n.val.ind){
  print(val.ind)
  mu.val = val.para.grid$mu[val.ind] ## \theta_1 in group 1
  sd.val = val.para.grid$sd[val.ind] ## \theta_1 in group 1
  delta.prop.val = val.para.grid$delta.prop[val.ind] ## proportion of group difference
  
  ## base treatment effect as in the training stage
  delta.val.base = qnorm(alpha, sd = sqrt(2*sd.val^2/n.sample.train), 
                         lower.tail = FALSE)-
    qnorm(0.8, sd = sqrt(2*sd.val^2/n.sample.train),lower.tail = FALSE)
  
  ## treatment effect for validation
  mu.val.trt = mu.val + delta.val.base*delta.prop.val
  val.para.grid$mu_trt[val.ind] = mu.val.trt
  
  ## generate validation data
  validation.mat = t(sapply(1:n.test.H0.outer.itt, function(x){
    ## simulate data for the first DNN
    data.val.fit = get.data.t.func(mu.grp.1.in = mu.val, 
                                   mu.grp.2.in = mu.val.trt, 
                                   sd.in = sd.val,
                                   n.in = n.sample.train,
                                   if.test = TRUE)
    
    ## return the data and p-values from other methods
    val.return.vec = c(data.val.fit$data,
                           data.val.fit$test)
    return(val.return.vec)
  }))
  
  ## write data to a data frame
  validation.frame = data.frame(validation.mat[, c(1:3)])
  colnames(validation.frame) = c("mean_diff", "grp1_sd", "grp2_sd")

  ## write p-values from other methods to a data frame
  validation.test.frame = data.frame(validation.mat[, c(4)])
  colnames(validation.test.frame) = c("t.test")
  
  ## get test statistics from the first DNN
  stats.val.vec = pred.DNN.normal(opt.model.weight, DNN.first.opt.layers, 
                             validation.frame,
                             col_means_train, col_stddevs_train)
  
  ## get pooled \theta_{12} under null
  sd.null = apply(validation.frame[, c("grp1_sd", "grp2_sd")], 1, mean)
  cutoff.val.data.input = data.frame("sd" = sd.null)
  
  ## calculate the corresponding critival value from the second DNN
  cutoff.val.vec = pred.DNN.normal(opt.model.cutoff.weight, DNN.cutoff.opt.layers, 
                                   cutoff.val.data.input,
                                  col_means_cutoff_train, 
                                  col_stddevs_cutoff_train)
  
  ## write type I error rates / power to the final output table
  val.para.grid[val.ind, c("DNN_power")] = mean(stats.val.vec>=cutoff.val.vec)
  val.para.grid[val.ind, c("t_power")] = mean(validation.test.frame$t.test<=alpha)

  print(val.para.grid)
  write.csv(val.para.grid, 
            paste0("sim_3/sim_3_n_", n.sample.train, ".csv"))
}

write.csv(val.para.grid, 
          paste0("sim_3/sim_3_n_", n.sample.train, ".csv"))















