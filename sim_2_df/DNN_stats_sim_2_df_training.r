
## This R code is to obtain the two traind DNNs for test statistics and critical
## values in the normal distribution with equal variance assumption
## of "DNN" in the second column of Table 2 of Section 4.2
## One can directly work on the validation code with the pre-trained DNN

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
library(MASS)
library(metRology)

###############################################################################
## set random seeds
set.seed(1)
## parameters
alpha = alpha.cutoff = 0.05 ## nominal one sided type I error
n.ind = 500 ## number of training features "A" in the first DNN
n.cutoff = 500 # same "A" as the size in the second DNN

## candidate theta in generating the training data
df.train.vec = runif(n.ind, min = 3, max = 10) # mean

n.sample.train = 200 ## sample size
n.cluster = 4 ## number of cores in parallel computing
max.epoch = 10^1 # number of training epochs in the first DNN
n.train.H0.itt = 1*10^4 # B_0 for null data in the first DNN
n.train.H1.itt = 1*10^4  # B_1 for alternative data in the second DNN
n.dim.data.train = 14 # dimension of t(s) for TS-DNN
n.train.itt = n.train.H0.itt + n.train.H1.itt

# number of iterations of calculating the critical value in the second DNN
n.test.H0.inner.itt = 10^5

###############################################################################
## training features for the first DNN
time.first.DNN = Sys.time()

###############################################################################
## generate training data for the first DNN
# for (ind in 1:n.ind){
cl = makeCluster(n.cluster)
registerDoParallel(cl)
data.train.para = foreach(ind=1:n.ind) %dopar% {  
  
  source("DNN_stats_functions.r")
  library(doParallel)
  library(keras)
  library(reticulate)
  library(tensorflow)
  library(keras)
  library(tibble)
  library(car)
  library(MASS)
  library(metRology)
  
  # print(paste("train ind:", ind))

  set.seed(ind)
  df.train = df.train.vec[ind] ## sd of the normal distribution
  
  ## difference between \theta_2 and \theta_2
  delta.train = 2
  
  df.trt.train = df.train + delta.train
  
  ## simulate null data
  data.train.H0 = t(sapply(1:n.train.H0.itt, 
       function(x){get.data.dist.t.func(df.grp.1.in = df.train, 
                                   df.grp.2.in = df.train, 
                                   n.in = n.sample.train,
                                   if.test = FALSE)$data}))
  
  ## simulate alternative data
  data.train.H1 = t(sapply(1:n.train.H1.itt, 
         function(x){get.data.dist.t.func(df.grp.1.in = df.train, 
                                     df.grp.2.in = df.trt.train, 
                                     n.in = n.sample.train,
                                     if.test = FALSE)$data}))

      ## aggregate training data
    data.train.pre = data.frame(rbind(data.train.H0, data.train.H1))
      ## labels for the training data
    data.train.pre$label = c(rep(0, n.train.H0.itt), rep(1, n.train.H1.itt))

    return(data.train.pre)

}
stopCluster(cl)

data.train = matrix(NA, nrow = (n.train.itt)*n.ind, 
                    ncol = n.dim.data.train)
data.train.label = rep(NA, n.ind)
for (ind in 1:n.ind){
  data.train.in.temp = unlist(data.train.para[[ind]])
  data.train.mat.temp = matrix(data.train.in.temp, 
                               nrow = n.train.itt,
                               ncol = n.dim.data.train+1, byrow = FALSE
  )
  data.train[(1:n.train.itt)+(ind-1)*n.train.itt, ] = 
    data.train.mat.temp[, 1:(n.dim.data.train)]
  data.train.label[(1:n.train.itt)+(ind-1)*n.train.itt] =
    as.numeric(data.train.mat.temp[, 1+n.dim.data.train])
}

############################################################################
## train the first DNN on test statistics
## normalize the inputs
data.train =  as_tibble(data.train)
data.train.scale =scale(data.train)

col_means_train <- attr(data.train.scale, "scaled:center")
col_stddevs_train <- attr(data.train.scale, "scaled:scale")

active.name = "relu" # activation function for the first DNN
## candidate DNN structure for cross-validation
DNN.first.nodes.cross = c(50, 100, 150, 50, 100, 150)
DNN.first.layers.cross = c(2, 2, 2, 3, 3, 3)
DNN.first.drop.cross = rep(0.1, 6)
DNN.first.mat = matrix(NA, nrow = length(DNN.first.nodes.cross), ncol = 4)

## test performane on all candidate structures
for (DNN.first.ind in 1:length(DNN.first.nodes.cross)){
  DNN.first.temp = DNN.fit(data.train.scale.in = data.train.scale, 
                           data.train.label.in = data.train.label, 
                           drop.rate.in = DNN.first.drop.cross[DNN.first.ind], 
                           active.name.in = active.name, 
                           n.node.in = DNN.first.nodes.cross[DNN.first.ind], 
                           n.layer.in = DNN.first.layers.cross[DNN.first.ind], 
                           max.epoch.in = max.epoch, validation.prop.in = 0.2)
  
  DNN.first.mat[DNN.first.ind, ] = 
    as.vector(unlist(lapply(DNN.first.temp$history$metrics,function(x){tail(x,1)})))
}

## select the DNN structure with the minimum validation loss
DNN.first.opt.nodes = DNN.first.nodes.cross[which.min(DNN.first.mat[, 4])]
DNN.first.opt.layers = DNN.first.layers.cross[which.min(DNN.first.mat[, 4])]
DNN.first.opt.drop = DNN.first.drop.cross[which.min(DNN.first.mat[, 4])]

## name of the final structure
first.DNN.name = paste0("nodes_", DNN.first.opt.nodes, 
                        "_layers_", DNN.first.opt.layers)

## fit the first DNN with all training data with the selected structure
DNN.first.opt = DNN.fit(data.train.scale.in = data.train.scale, 
                        data.train.label.in = data.train.label, 
                        drop.rate.in = DNN.first.opt.drop, 
                        active.name.in = active.name, 
                        n.node.in = DNN.first.opt.nodes, 
                        n.layer.in = DNN.first.opt.layers, 
                        max.epoch.in = max.epoch, validation.prop.in = 0)

## get the DNN weights
opt.model.weight = get_weights(DNN.first.opt$model)
print(DNN.first.opt)

## save TS-DNN to files
save_model_hdf5(DNN.first.opt$model, 
                "sim_2_df/sim_2_first_DNN", 
                overwrite = TRUE, include_optimizer = TRUE)
time.first.DNN = Sys.time() - time.first.DNN

##########################################################################
## generate training data for the second DNN to approximate critical values
time.second.DNN = Sys.time()
df.cutoff.vec = df.train.vec[1:n.cutoff]

## parallel computing
cl = makeCluster(n.cluster)
registerDoParallel(cl)

## simulate training data for the second DNN
para.cutoff.fit = foreach(cutoff.ind=1:n.cutoff) %dopar% {

  source("DNN_stats_functions.r")
  library(keras)
  library(reticulate)
  library(tensorflow)
  library(keras)
  library(tibble)
  library(car)
  library(MASS)
  library(metRology)
  
  df.cutoff = df.cutoff.vec[cutoff.ind] ## common sd under null

  ## simulate null data
  data.cutoff.H0 = t(sapply(1:n.test.H0.inner.itt, 
         function(x){get.data.dist.t.func(df.grp.1.in = df.cutoff, 
                                     df.grp.2.in = df.cutoff, 
                                     n.in = n.sample.train,
                                     if.test = FALSE)$data}))
  
  ## get predictive test statistics from the first DNN
  null.cutoff.pred = pred.DNN.normal(opt.model.weight, DNN.first.opt.layers, 
                                     data.cutoff.H0,
                                   col_means_train, col_stddevs_train)
  
  ## output upper working alpha quantile to control the type I error
  return(quantile(null.cutoff.pred, prob = 1-alpha.cutoff, type=3))
}
stopCluster(cl)

##################################################################################
## normalize inputs for the second DNN
data.cutoff.train = data.frame("df" = df.cutoff.vec)
data.cutoff.train =  as_tibble(data.cutoff.train)
data.cutoff.train.scale =scale(data.cutoff.train)

col_means_cutoff_train <- attr(data.cutoff.train.scale, "scaled:center")
col_stddevs_cutoff_train <- attr(data.cutoff.train.scale, "scaled:scale")

## save scale parameters to a file
col_mean_sd_two = list("col_means_train" = col_means_train, 
                       "col_stddevs_train" = col_stddevs_train,
                       "col_means_cutoff_train" = col_means_cutoff_train,
                       "col_stddevs_cutoff_train" = col_stddevs_cutoff_train
)

save(col_mean_sd_two, file = "sim_2_df/sim_2_scale_parameters")

## cross validation on the structure of the second DNN
DNN.cutoff.nodes.cross = c(50, 100, 150, 50, 100, 150)
DNN.cutoff.layers.cross = c(2, 2, 2, 3, 3, 3)
DNN.cross.mat = matrix(NA, nrow = length(DNN.cutoff.nodes.cross), ncol = 4)

for (cross.ind in 1:length(DNN.cutoff.nodes.cross)){
  DNN.cal.cutoff.cross = DNN.cutoff.fit(data.cutoff.train.scale, 
                                        unlist(para.cutoff.fit),
                                      0.1, "relu", DNN.cutoff.nodes.cross[cross.ind], 
                                      DNN.cutoff.layers.cross[cross.ind], 
                                      1*10^3, 0.2)
  DNN.cross.mat[cross.ind, ] = 
    as.vector(unlist(lapply(DNN.cal.cutoff.cross$history$metrics,function(x){tail(x,1)})))
}

## select the one with minimum validation loss
DNN.cutoff.opt.nodes = DNN.cutoff.nodes.cross[which.min(DNN.cross.mat[, 4])]
DNN.cutoff.opt.layers = DNN.cutoff.layers.cross[which.min(DNN.cross.mat[, 4])]

## first second DNN with all training data
DNN.cal.cutoff.fit = DNN.cutoff.fit(data.cutoff.train.scale, 
                                    unlist(para.cutoff.fit),
                                      0.1, "relu", DNN.cutoff.opt.nodes,
                                    DNN.cutoff.opt.layers, 
                                    1*10^3, 0)

second.DNN.name = paste0("nodes_", DNN.cutoff.opt.nodes, 
                         "_layers_", DNN.cutoff.opt.layers)

pred = DNN.cal.cutoff.fit$model %>% predict(data.cutoff.train.scale)
## get the DNN weights
opt.model.cutoff.weight = get_weights(DNN.cal.cutoff.fit$model)
print(DNN.cal.cutoff.fit$history)

## save CV-DNN to files
save_model_hdf5(DNN.cal.cutoff.fit$model, 
                "sim_2_df/sim_2_second_DNN", 
                overwrite = TRUE, include_optimizer = TRUE)

time.second.DNN = Sys.time() - time.second.DNN

#############################################################################
## DNN.time
DNN.time.out = matrix(c(time.first.DNN, time.second.DNN))
rownames(DNN.time.out) = c("DNN.first.time", "DNN.second.time")
write.csv(DNN.time.out, file = "sim_2_df/DNN_training_time.csv")








