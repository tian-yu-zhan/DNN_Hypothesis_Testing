
## R code for validation to reproduce results of "DNN" in the second column of
## Table 2 in Section 4.2 with pre-trained DNNs

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
n.sample.train = 200 ## sample size
n.test.H0.outer.itt = 10^6 # number of iterations of validating type I error and power
n.cluster = 4 # number of clusters in parallel computing

###################################################################################
## first DNN on test statistics
first.DNN.model = load_model_hdf5("sim_2_df/sim_2_first_DNN", 
                                  custom_objects = NULL, compile = TRUE)
## get the DNN weights
opt.model.weight = get_weights(first.DNN.model)
## get the number of layers 
DNN.first.opt.layers = length(opt.model.weight)/2-1

####################################################################################
## second DNN on critical values
second.DNN.model = load_model_hdf5("sim_2_df/sim_2_second_DNN", 
                                   custom_objects = NULL, compile = TRUE)
## get the DNN weights
opt.model.cutoff.weight = get_weights(second.DNN.model)
## get the number of layers 
DNN.cutoff.opt.layers = length(opt.model.cutoff.weight)/2-1

###################################################################################
## raed scale parameters
load(file = "sim_2_df/sim_2_scale_parameters")

col_means_train = col_mean_sd_two$col_means_train
col_stddevs_train = col_mean_sd_two$col_stddevs_train
col_means_cutoff_train = col_mean_sd_two$col_means_cutoff_train
col_stddevs_cutoff_train = col_mean_sd_two$col_stddevs_cutoff_train


####################################################################################
## validation of type I error rates and power 
val.para.grid = data.frame("df.1" = rep(c(4, 7), each = 4))
val.para.grid$df.2 = val.para.grid$df.1 + rep(c(0, 1, 2, 3), times = 2)


val.para.grid$bartlett_power = val.para.grid$levene_power = val.para.grid$fligner_power = 
val.para.grid$LRT_power = 
  val.para.grid$DNN_power = NA
n.val.ind = dim(val.para.grid)[1]

## evaluate each scenario
for (val.ind in c(1:n.val.ind)){
  print(val.ind)
  
  df.1.val = val.para.grid$df.1[val.ind]
  df.2.val = val.para.grid$df.2[val.ind]
  
  ## generate validation data with parallel computing
  cl = makeCluster(n.cluster)
  registerDoParallel(cl)
  data.val.fit.para = foreach(test.H0.outer.itt=1:n.test.H0.outer.itt) %dopar% {
    
    ## random seed for parallel computing
    set.seed(n.test.H0.outer.itt*val.ind + test.H0.outer.itt)
    
    source("DNN_stats_functions.r")
    library(keras)
    library(reticulate)
    library(tensorflow)
    library(keras)
    library(tibble)
    library(car)
    library(MASS)
    library(metRology)
    
    ## simulate data for the first DNN
    data.val.fit.temp = get.data.dist.t.func(df.grp.1.in = df.1.val, 
                                        df.grp.2.in = df.2.val, 
                                        n.in = n.sample.train,
                                        if.test = TRUE)
    
    ## return the data and p-values from other methods
    val.return.vec = c(data.val.fit.temp$data,
                       data.val.fit.temp$mle,
                       data.val.fit.temp$test
    )
    return(val.return.vec)
  }
  stopCluster(cl)
  
  validation.mat = matrix(unlist(data.val.fit.para), nrow = n.test.H0.outer.itt,
                        ncol = 19, byrow = TRUE)
  
  ## write data to a data frame
  validation.frame = data.frame(validation.mat[, c(1:14)])
  # colnames(validation.frame) = c("min_1", "first_1", "med_1", "mean_1", 
  #                                "third_1", "max_1", "sd_1",
  #                                "min_2", "first_2", "med_2", "mean_2", 
  #                                "third_2", "max_2", "sd_2"
  #                                )

  ## write p-values from other methods to a data frame
  validation.test.frame = data.frame(validation.mat[, c(16:19)])
  colnames(validation.test.frame) = c("LRT.test", "bartlett.test", 
                                      "levene.test", "fligner.test")
  
  ## get test statistics from the first DNN
  stats.val.vec = pred.DNN.normal(opt.model.weight, DNN.first.opt.layers, 
                             validation.frame,
                             col_means_train, col_stddevs_train)
  
  ## get pooled \theta_{12} under null
  # mu.null = apply(validation.frame[, c("grp1_mean", "grp2_mean")], 1, mean)
  df.null = as.numeric(validation.mat[, c(15)])
  cutoff.val.data.input = data.frame("df" = df.null)
  
  ## calculate the corresponding critival value from the second DNN
  cutoff.val.vec = pred.DNN.normal(opt.model.cutoff.weight, DNN.cutoff.opt.layers, 
                                   cutoff.val.data.input,
                                  col_means_cutoff_train, 
                                  col_stddevs_cutoff_train)
  
  ## write type I error rates / power to the final output table
  val.para.grid[val.ind, c("DNN_power")] = mean(stats.val.vec>=cutoff.val.vec)
  val.para.grid[val.ind, c("LRT_power")] = mean(validation.test.frame$LRT.test<=alpha)
  val.para.grid[val.ind, c("bartlett_power")] = 
    mean(validation.test.frame$bartlett.test<=alpha)
  val.para.grid[val.ind, c("levene_power")] = 
    mean(validation.test.frame$levene.test<=alpha)
  val.para.grid[val.ind, c("fligner_power")] = 
    mean(validation.test.frame$fligner.test<=alpha)

  print(val.para.grid)
  write.csv(val.para.grid, 
            paste0("sim_2_df/sim_2_n_", n.sample.train, ".csv"))
}

write.csv(val.para.grid, 
          paste0("sim_2_df/sim_2_n_", n.sample.train, ".csv"))















