##############################################################################
## sim 1 of scale uniform distribution
quiet <- function(x) { 
  sink(tempfile()) 
  on.exit(sink()) 
  invisible(force(x)) 
} 

## function to get the training data for the first DNN
get.data.unif.func = function(theta.grp.1.in, 
                              theta.grp.2.in, 
                              k.in, 
                              n.in,
                              if.test,
                              if.MMD){
  
  ## simulate data
  data.grp.1.in = runif(n.in, min = (1-k.in)*theta.grp.1.in, 
                        max =  (1+k.in)*theta.grp.1.in)
  # data.grp.1.summary = c(summary(data.grp.1.in), sd(data.grp.1.in))
  data.grp.1.summary = c(min(data.grp.1.in), max(data.grp.1.in))
  
  data.grp.2.in = runif(n.in, min = (1-k.in)*theta.grp.2.in, 
                        max =  (1+k.in)*theta.grp.2.in)
  # data.grp.2.summary = c(summary(data.grp.2.in), sd(data.grp.2.in))
  data.grp.2.summary = c(min(data.grp.2.in), max(data.grp.2.in))
  
  data.grp.12.in = c(data.grp.1.in, data.grp.2.in)
  
  ## if add t test and wilcox test
  if (if.test){
    t.test.p.value = t.test(x = data.grp.2.in, y = data.grp.1.in, 
                            alternative = "greater")$p.value
    wilcox.p.value = wilcox.test(x = data.grp.2.in, y = data.grp.1.in, 
                            alternative = "greater")$p.value
    ## LRT test statistics
    LRT.stats = max(data.grp.2.in)/max(data.grp.1.in)
  } else{
    t.test.p.value = wilcox.p.value = LRT.stats = -1000
  }
  
  ## if conduct MMD test
  if (if.MMD){
    ## MMD test
    kmmd.fit = quiet(
      kmmd(matrix(data.grp.1.in), matrix(data.grp.2.in),
           kernel="rbfdot",kpar="automatic", alpha = alpha,
           asymptotic = TRUE, replace = TRUE, ntimes = 150, frac = 1))
    kmmd.dec = kmmd.fit@AsympH0
  } else {
    kmmd.dec = -1000
  }
  
  new.list = list("data" = c(data.grp.1.summary, data.grp.2.summary, k.in),
                  "mean" = mean(c(data.grp.1.in, data.grp.2.in)), 
                  "test" = c(t.test.p.value, wilcox.p.value, kmmd.dec, LRT.stats))
  return(new.list)
}

## T2 statistics from reviewer 1
sim.1.T2.func = function(min.1.in, max.1.in, min.2.in, max.2.in, k.in){
  w = (1-k.in)^2/((1-k.in)^2+(1+k.in)^2)
  T2 = (w*min.2.in/(1-k.in)+(1-w)*max.2.in/(1+k.in))/
    (w*min.1.in/(1-k.in)+(1-w)*max.1.in/(1+k.in))
}

## function to get the training data for the first DNN with T2
get.data.unif.T2.func = function(theta.grp.1.in, 
                              theta.grp.2.in, 
                              k.in, 
                              n.in,
                              if.test,
                              if.MMD){
  
  ## simulate data
  data.grp.1.in = runif(n.in, min = (1-k.in)*theta.grp.1.in, 
                        max =  (1+k.in)*theta.grp.1.in)
  # data.grp.1.summary = c(summary(data.grp.1.in), sd(data.grp.1.in))
  data.grp.1.summary = c(min(data.grp.1.in), max(data.grp.1.in))
  
  data.grp.2.in = runif(n.in, min = (1-k.in)*theta.grp.2.in, 
                        max =  (1+k.in)*theta.grp.2.in)
  # data.grp.2.summary = c(summary(data.grp.2.in), sd(data.grp.2.in))
  data.grp.2.summary = c(min(data.grp.2.in), max(data.grp.2.in))
  
  data.grp.12.in = c(data.grp.1.in, data.grp.2.in)
  
  T2.stats = sim.1.T2.func(min.1.in = min(data.grp.1.in), 
                           max.1.in = max(data.grp.1.in), 
                           min.2.in = min(data.grp.2.in), 
                           max.2.in = max(data.grp.2.in), 
                           k.in = k.in)
  
  ## if add t test and wilcox test
  if (if.test){
    t.test.p.value = t.test(x = data.grp.2.in, y = data.grp.1.in, 
                            alternative = "greater")$p.value
    wilcox.p.value = wilcox.test(x = data.grp.2.in, y = data.grp.1.in, 
                                 alternative = "greater")$p.value
    ## LRT test statistics
    LRT.stats = max(data.grp.2.in)/max(data.grp.1.in)
  } else{
    t.test.p.value = wilcox.p.value = LRT.stats = -1000
  }
  
  ## if conduct MMD test
  if (if.MMD){
    ## MMD test
    kmmd.fit = quiet(
      kmmd(matrix(data.grp.1.in), matrix(data.grp.2.in),
           kernel="rbfdot",kpar="automatic", alpha = alpha,
           asymptotic = TRUE, replace = TRUE, ntimes = 150, frac = 1))
    kmmd.dec = kmmd.fit@AsympH0
  } else {
    kmmd.dec = -1000
  }
  
  new.list = list("data" = c(data.grp.1.summary, data.grp.2.summary, k.in, T2.stats),
                  "mean" = mean(c(data.grp.1.in, data.grp.2.in)), 
                  "test" = c(t.test.p.value, wilcox.p.value, kmmd.dec, LRT.stats,
                             T2.stats))
  return(new.list)
}

## function to get the training data for the first DNN with summary statistics
get.data.unif.summary.func = function(theta.grp.1.in, 
                              theta.grp.2.in, 
                              k.in, 
                              n.in,
                              if.test,
                              if.MMD){
  
  ## simulate data
  data.grp.1.in = runif(n.in, min = (1-k.in)*theta.grp.1.in, 
                        max =  (1+k.in)*theta.grp.1.in)
  # data.grp.1.summary = c(summary(data.grp.1.in), sd(data.grp.1.in))
  data.grp.1.summary = c(summary(data.grp.1.in), sd(data.grp.1.in))
  
  data.grp.2.in = runif(n.in, min = (1-k.in)*theta.grp.2.in, 
                        max =  (1+k.in)*theta.grp.2.in)
  # data.grp.2.summary = c(summary(data.grp.2.in), sd(data.grp.2.in))
  data.grp.2.summary =c(summary(data.grp.2.in), sd(data.grp.2.in))
  
  data.grp.12.in = c(data.grp.1.in, data.grp.2.in)
  
  ## if add t test and wilcox test
  if (if.test){
    t.test.p.value = t.test(x = data.grp.2.in, y = data.grp.1.in, 
                            alternative = "greater")$p.value
    wilcox.p.value = wilcox.test(x = data.grp.2.in, y = data.grp.1.in, 
                                 alternative = "greater")$p.value
    ## LRT test statistics
    LRT.stats = max(data.grp.2.in)/max(data.grp.1.in)
  } else{
    t.test.p.value = wilcox.p.value = LRT.stats = -1000
  }
  
  ## if conduct MMD test
  if (if.MMD){
    ## MMD test
    kmmd.fit = quiet(
      kmmd(matrix(data.grp.1.in), matrix(data.grp.2.in),
           kernel="rbfdot",kpar="automatic", alpha = alpha,
           asymptotic = TRUE, replace = TRUE, ntimes = 150, frac = 1))
    kmmd.dec = kmmd.fit@AsympH0
  } else {
    kmmd.dec = -1000
  }
  
  new.list = list("data" = c(data.grp.1.summary, data.grp.2.summary, k.in),
                  "mean" = mean(c(data.grp.1.in, data.grp.2.in)), 
                  "test" = c(t.test.p.value, wilcox.p.value, kmmd.dec, LRT.stats))
  return(new.list)
}

##############################################################################
## sim 2 of testing degrees of freedom in t-distribution
t.opt = function(x, data){
  -sum(log(dt(data, df = x)))
}

## with summary statistics
get.data.dist.t.func = function(df.grp.1.in,
                                df.grp.2.in,
                                n.in,
                                if.test){
  
  ## simulate data
  data.grp.1.in = rt.scaled(n = n.in, df=df.grp.1.in)
  # data.grp.1.in = rcauchy(n.in, location = mu.grp.1.in, scale = sd.1.in)
  data.grp.1.summary = c(summary(data.grp.1.in), sd(data.grp.1.in))
  
  data.grp.2.in = rt.scaled(n = n.in, df=df.grp.2.in)
  # data.grp.2.in = rcauchy(n.in, location = mu.grp.2.in, scale = sd.2.in)
  data.grp.2.summary = c(summary(data.grp.2.in), sd(data.grp.2.in))
  
  data.grp.12.in = c(data.grp.1.in, data.grp.2.in)
  
  data.out = c(data.grp.1.summary, data.grp.2.summary)
  
  ## if add t test and wilcox test
  if (if.test){
    ## dataset for bartlett test and Levene’s test
    data.test = data.frame("x" = c(data.grp.1.in, data.grp.2.in),
                           "grp" = factor(rep(1:2, each = n.in)))
    
    bartlett.test.p.value = bartlett.test.func(data.test.inn = data.test,
                                               data.grp.1.inn = data.grp.1.in,
                                               data.grp.2.inn = data.grp.2.in)
    
    levene.test.p.value = levene.test.func(data.test.inn = data.test,
                                           data.grp.1.inn = data.grp.1.in,
                                           data.grp.2.inn = data.grp.2.in)
    
    fligner.test.p.value = fligner.test.func(data.test.inn = data.test,
                                             data.grp.1.inn = data.grp.1.in,
                                             data.grp.2.inn = data.grp.2.in)
    
    
    ## likelihood ratio test
    ## calculate MLE
    MLE.12.fit = optimize(t.opt, data = data.grp.12.in, interval = c(3, 10))$minimum
    MLE.1.fit = optimize(t.opt, data = data.grp.1.in, interval = c(3, 10))$minimum
    MLE.2.fit = optimize(t.opt, data = data.grp.2.in, interval = c(3, 10))$minimum
    
    MLE.12 = as.numeric(unlist(MLE.12.fit))[1]
    MLE.1 = as.numeric(unlist(MLE.1.fit))[1]
    MLE.2 = as.numeric(unlist(MLE.2.fit))[1]
    
    MLE.out.vec = MLE.12
    
    log.like.rest = sum(log(dt.scaled(data.grp.12.in, df=MLE.12)))
    log.like.full = sum(log(dt.scaled(data.grp.1.in,df=MLE.1)))+
      sum(log(dt.scaled(data.grp.2.in, df=MLE.2)))
    
    LRT.stats = 2*(log.like.full-log.like.rest)
    
    # LRT.p.value = pchisq(2*(log.like.full-log.like.rest), df=1, lower.tail = FALSE)
    # two-sided test LRT test
    if (MLE.2>MLE.1){
      LRT.p.value = pchisq(LRT.stats, df=1, lower.tail = FALSE)/2
    } else {
      LRT.p.value = 1-pchisq(LRT.stats, df=1, lower.tail = FALSE)/2
    }
    
  } else{
    bartlett.test.p.value = levene.test.p.value = fligner.test.p.value = 
      MLE.out.vec = LRT.stats = LRT.p.value = NULL
  }
  
  new.list = list(
    "mle" = MLE.out.vec,
    "data" = data.out,
    "test" = c(LRT.p.value, bartlett.test.p.value, levene.test.p.value,
               fligner.test.p.value)
  )
  return(new.list)
}

## with summary statistics and LRT
get.data.dist.t.LRT.func = function(df.grp.1.in,
                                df.grp.2.in,
                                n.in,
                                if.test){
  
  ## simulate data
  data.grp.1.in = rt.scaled(n = n.in, df=df.grp.1.in)
  # data.grp.1.in = rcauchy(n.in, location = mu.grp.1.in, scale = sd.1.in)
  data.grp.1.summary = c(summary(data.grp.1.in), sd(data.grp.1.in))
  
  data.grp.2.in = rt.scaled(n = n.in, df=df.grp.2.in)
  # data.grp.2.in = rcauchy(n.in, location = mu.grp.2.in, scale = sd.2.in)
  data.grp.2.summary = c(summary(data.grp.2.in), sd(data.grp.2.in))
  
  data.grp.12.in = c(data.grp.1.in, data.grp.2.in)
  
  ## calculate MLE
  MLE.12.fit = optimize(t.opt, data = data.grp.12.in, interval = c(3, 10))$minimum
  MLE.1.fit = optimize(t.opt, data = data.grp.1.in, interval = c(3, 10))$minimum
  MLE.2.fit = optimize(t.opt, data = data.grp.2.in, interval = c(3, 10))$minimum
  
  MLE.12 = as.numeric(unlist(MLE.12.fit))[1]
  MLE.1 = as.numeric(unlist(MLE.1.fit))[1]
  MLE.2 = as.numeric(unlist(MLE.2.fit))[1]
  
  MLE.out.vec = MLE.12
  
  log.like.rest = sum(log(dt.scaled(data.grp.12.in, df=MLE.12)))
  log.like.full = sum(log(dt.scaled(data.grp.1.in,df=MLE.1)))+
    sum(log(dt.scaled(data.grp.2.in, df=MLE.2)))
  
  LRT.stats = 2*(log.like.full-log.like.rest)
  
  data.out = c(data.grp.1.summary, data.grp.2.summary, LRT.stats)
  
  ## if add t test and wilcox test
  if (if.test){
    ## dataset for bartlett test and Levene’s test
    data.test = data.frame("x" = c(data.grp.1.in, data.grp.2.in),
                           "grp" = factor(rep(1:2, each = n.in)))
    
    bartlett.test.p.value = bartlett.test.func(data.test.inn = data.test,
                                               data.grp.1.inn = data.grp.1.in,
                                               data.grp.2.inn = data.grp.2.in)
    
    levene.test.p.value = levene.test.func(data.test.inn = data.test,
                                           data.grp.1.inn = data.grp.1.in,
                                           data.grp.2.inn = data.grp.2.in)
    
    fligner.test.p.value = fligner.test.func(data.test.inn = data.test,
                                             data.grp.1.inn = data.grp.1.in,
                                             data.grp.2.inn = data.grp.2.in)
    
    
    ## likelihood ratio test
    if (MLE.2>MLE.1){
      LRT.p.value = pchisq(LRT.stats, df=1, lower.tail = FALSE)/2
    } else {
      LRT.p.value = 1-pchisq(LRT.stats, df=1, lower.tail = FALSE)/2
    }
    
  } else{
    bartlett.test.p.value = levene.test.p.value = fligner.test.p.value = 
      MLE.out.vec = LRT.p.value = NULL
  }
  
  new.list = list(
    "mle" = MLE.out.vec,
    "data" = data.out,
    "test" = c(LRT.p.value, bartlett.test.p.value, levene.test.p.value,
               fligner.test.p.value)
  )
  return(new.list)
}

## Bartlett test for testing variance (inflated type I error)
bartlett.test.func = function(data.test.inn, data.grp.1.inn, data.grp.2.inn){
  two.sided.test = bartlett.test(x ~ grp, data = data.test.inn)
  if (var(data.grp.2.inn)<var(data.grp.1.inn)){
    one.sided.p.value =  two.sided.test$p.value/2
  } else {
    one.sided.p.value =  1-two.sided.test$p.value/2
  }

  return(one.sided.p.value)
}

## Fligner test for testing variance
fligner.test.func = function(data.test.inn, data.grp.1.inn, data.grp.2.inn){
  two.sided.test = fligner.test(x ~ grp, data = data.test.inn)
  if (var(data.grp.2.inn)<var(data.grp.1.inn)){
    one.sided.p.value =  two.sided.test$p.value/2
  } else {
    one.sided.p.value =  1-two.sided.test$p.value/2
  }
  
  return(one.sided.p.value)
}

## Levene test for testing variance
levene.test.func = function(data.test.inn, data.grp.1.inn, data.grp.2.inn){
  two.sided.test = leveneTest(x ~ grp, data = data.test.inn)
  if (var(data.grp.2.inn)<var(data.grp.1.inn)){
    one.sided.p.value =  two.sided.test$`Pr(>F)`[1]/2
  } else {
    one.sided.p.value =  1-two.sided.test$`Pr(>F)`[1]/2
  }
  return(one.sided.p.value)
}


###############################################################################
## simulation 3 on two-sample t-test
get.data.t.func = function(mu.grp.1.in, 
                           mu.grp.2.in, 
                           sd.in,
                           n.in,
                          if.test){
  
  ## simulate data
  data.grp.1.in = rnorm(n.in, mean = mu.grp.1.in, sd = sd.in)
  data.grp.1.summary = c(mean(data.grp.1.in), sd(data.grp.1.in))
  
  data.grp.2.in = rnorm(n.in, mean = mu.grp.2.in, sd = sd.in)
  data.grp.2.summary = c(mean(data.grp.2.in), sd(data.grp.2.in))
  
  data.grp.12.in = c(data.grp.1.in, data.grp.2.in)
  
  ## if add t test and wilcox test
  if (if.test){
    t.test.p.value = t.test(x = data.grp.2.in, y = data.grp.1.in, var.equal = TRUE,
                            alternative = "greater")$p.value
    
  } else{
    t.test.p.value = NULL
  }
  
  new.list = list(
                  # "data" = c(data.grp.1.summary, data.grp.2.summary),
                  "data" = c(mean(data.grp.2.in) - mean(data.grp.1.in),
                             sd(data.grp.1.in), sd(data.grp.2.in)),
                  "test" = c(t.test.p.value)
  )
  return(new.list)
}

#############################################################################
## case study of adaptive designs

## function to get the training data for the first DNN
get.data.case.func = function(prop.grp.1.train.in, 
                                 prop.grp.2.train.in, 
                                 n.in, 
                                 if.test){
  
  ## simulate first stage data
  data.grp.1 = rbinom(n.in, size = 1, prob = prop.grp.1.train.in)
  data.grp.2 = rbinom(n.in, size = 1, prob = prop.grp.2.train.in)

  adap.ind = (mean(data.grp.2) - mean(data.grp.1))>n.2.prop.cutoff
  if (adap.ind){
    n.2.in = n.2.min
  } else{
    n.2.in = n.2.max
  }
  
  ## simulate second stage data
  data.grp.1.stage.2 = rbinom(n.2.in, size = 1, prob = prop.grp.1.train.in)
  data.grp.2.stage.2 = rbinom(n.2.in, size = 1, prob = prop.grp.2.train.in)
  
  data.return.vec = c(mean(data.grp.1),
                      mean(data.grp.2),
                      mean(data.grp.1.stage.2),
                      mean(data.grp.2.stage.2),
                      n.2.in
  )
  
  ## if add t test and wilcox test
  if (if.test){
    naive.p.value = prop.test(x = c(max(1, sum(data.grp.1)+sum(data.grp.1.stage.2)), 
                             max(1, sum(data.grp.2)+sum(data.grp.2.stage.2))),
                       n = rep(n.in+n.2.in, 2),
                       alternative = "less",
                       correct = FALSE)$p.value
    
    p.1.value = prop.test(x = c(max(1, sum(data.grp.1)), 
                                    max(1, sum(data.grp.2))),
                              n = rep(n.in, 2),
                              alternative = "less",
                              correct = FALSE)$p.value
    
    p.2.value = prop.test(x = c(max(1, sum(data.grp.1.stage.2)), 
                                    max(1, sum(data.grp.2.stage.2))),
                              n = rep(n.2.in, 2),
                              alternative = "less",
                              correct = FALSE)$p.value
    
    w1 = 1/2
    comb.p.value = pnorm(sqrt(w1)*qnorm(p.1.value, lower.tail = FALSE)+
                       sqrt(1-w1)*qnorm(p.2.value, lower.tail = FALSE), 
                       lower.tail = FALSE)
    
  } else{
    comb.p.value = naive.p.value = NULL
  }
  
  new.list = list("data" = data.return.vec,
                  "test" = c(comb.p.value, naive.p.value))
  return(new.list)
}

## function to get the training data for CV-DNN with statistic T
get.data.case.T.func = function(prop.grp.1.train.in, 
                              prop.grp.2.train.in, 
                              n.in, 
                              if.test){
  
  ## simulate first stage data
  data.grp.1 = rbinom(n.in, size = 1, prob = prop.grp.1.train.in)
  data.grp.2 = rbinom(n.in, size = 1, prob = prop.grp.2.train.in)
  
  adap.ind = (mean(data.grp.2) - mean(data.grp.1))>n.2.prop.cutoff
  if (adap.ind){
    n.2.in = n.2.min
  } else{
    n.2.in = n.2.max
  }
  
  ## simulate second stage data
  data.grp.1.stage.2 = rbinom(n.2.in, size = 1, prob = prop.grp.1.train.in)
  data.grp.2.stage.2 = rbinom(n.2.in, size = 1, prob = prop.grp.2.train.in)
  
  ## calculate T statistic
  w.T.out = sqrt(n.in)/(sqrt(n.in)+sqrt(n.2.in))
  T.stats.out = w.T.out*(mean(data.grp.2) - mean(data.grp.1)) + 
    (1-w.T.out)*(mean(data.grp.2.stage.2) - mean(data.grp.1.stage.2))
  
  data.return.vec = T.stats.out
  
  ## if add t test and wilcox test
  if (if.test){
    naive.p.value = prop.test(x = c(max(1, sum(data.grp.1)+sum(data.grp.1.stage.2)), 
                                    max(1, sum(data.grp.2)+sum(data.grp.2.stage.2))),
                              n = rep(n.in+n.2.in, 2),
                              alternative = "less",
                              correct = FALSE)$p.value
    
    p.1.value = prop.test(x = c(max(1, sum(data.grp.1)), 
                                max(1, sum(data.grp.2))),
                          n = rep(n.in, 2),
                          alternative = "less",
                          correct = FALSE)$p.value
    
    p.2.value = prop.test(x = c(max(1, sum(data.grp.1.stage.2)), 
                                max(1, sum(data.grp.2.stage.2))),
                          n = rep(n.2.in, 2),
                          alternative = "less",
                          correct = FALSE)$p.value
    
    w1 = 1/2
    comb.p.value = pnorm(sqrt(w1)*qnorm(p.1.value, lower.tail = FALSE)+
                           sqrt(1-w1)*qnorm(p.2.value, lower.tail = FALSE), 
                         lower.tail = FALSE)
    
  } else{
    comb.p.value = naive.p.value = NULL
  }
  
  new.list = list("data" = data.return.vec,
                  "test" = c(comb.p.value, naive.p.value),
                  "mle" = mean(c(mean(data.grp.2), mean(data.grp.1)))
                  )
  return(new.list)
}

## function to get the training data with statistic T
get.data.case.T.DNN.func = function(prop.grp.1.train.in, 
                                prop.grp.2.train.in, 
                                n.in, 
                                if.test){
  
  ## simulate first stage data
  data.grp.1 = rbinom(n.in, size = 1, prob = prop.grp.1.train.in)
  data.grp.2 = rbinom(n.in, size = 1, prob = prop.grp.2.train.in)
  
  adap.ind = (mean(data.grp.2) - mean(data.grp.1))>n.2.prop.cutoff
  if (adap.ind){
    n.2.in = n.2.min
  } else{
    n.2.in = n.2.max
  }
  
  ## simulate second stage data
  data.grp.1.stage.2 = rbinom(n.2.in, size = 1, prob = prop.grp.1.train.in)
  data.grp.2.stage.2 = rbinom(n.2.in, size = 1, prob = prop.grp.2.train.in)
  
  ## calculate T statistic
  w.T.out = sqrt(n.in)/(sqrt(n.in)+sqrt(n.2.in))
  T.stats.out = w.T.out*(mean(data.grp.2) - mean(data.grp.1)) + 
    (1-w.T.out)*(mean(data.grp.2.stage.2) - mean(data.grp.1.stage.2))
  T.p.value.out = pnorm(T.stats.out, lower.tail = FALSE)
  
  data.return.vec = c(mean(data.grp.1),
                      mean(data.grp.2),
                      mean(data.grp.1.stage.2),
                      mean(data.grp.2.stage.2),
                      n.2.in,
                      T.stats.out
  )
  
  ## if add t test and wilcox test
  if (if.test){
    naive.p.value = prop.test(x = c(max(1, sum(data.grp.1)+sum(data.grp.1.stage.2)), 
                                    max(1, sum(data.grp.2)+sum(data.grp.2.stage.2))),
                              n = rep(n.in+n.2.in, 2),
                              alternative = "less",
                              correct = FALSE)$p.value
    
    p.1.value = prop.test(x = c(max(1, sum(data.grp.1)), 
                                max(1, sum(data.grp.2))),
                          n = rep(n.in, 2),
                          alternative = "less",
                          correct = FALSE)$p.value
    
    p.2.value = prop.test(x = c(max(1, sum(data.grp.1.stage.2)), 
                                max(1, sum(data.grp.2.stage.2))),
                          n = rep(n.2.in, 2),
                          alternative = "less",
                          correct = FALSE)$p.value
    
    w1 = 1/2
    comb.p.value = pnorm(sqrt(w1)*qnorm(p.1.value, lower.tail = FALSE)+
                           sqrt(1-w1)*qnorm(p.2.value, lower.tail = FALSE), 
                         lower.tail = FALSE)
    
  } else{
    comb.p.value = naive.p.value = NULL
  }
  
  new.list = list("data" = data.return.vec,
                  "test" = c(comb.p.value, naive.p.value, T.p.value.out)
  )
  return(new.list)
}




############################################################################
## function to obtain statistics from the linear predictor of the first DNN
pred.DNN.normal = function(DNN.final.weights.in, n.layer.in, data.train.in,
                           col_means_train.in, col_stddevs_train.in){
  
  w1.scale = DNN.final.weights.in[[1]]
  b1.scale = as.matrix(DNN.final.weights.in[[2]])
  w1 = t(w1.scale/matrix(rep(col_stddevs_train.in, dim(w1.scale)[2]),
                         nrow = dim(w1.scale)[1], ncol = dim(w1.scale)[2]))
  b1 = b1.scale - t(w1.scale)%*%as.matrix(col_means_train.in/col_stddevs_train.in)
  
  for (wb.itt in 2:(n.layer.in+1)){
    w.text = paste0("w", wb.itt, "=t(DNN.final.weights.in[[", wb.itt*2-1, "]])")
    b.text = paste0("b", wb.itt, "= as.matrix(DNN.final.weights.in[[", wb.itt*2, "]])")
    
    eval(parse(text=w.text))
    eval(parse(text=b.text))
  }
  
  eval_f_whole_text1 = paste0(
    "eval_f <- function( x ) {x.mat = as.matrix(as.numeric(x), nrow = length(x), ncol = 1);
    w1x = (w1)%*%x.mat + b1;sw1x = as.matrix(c(relu(w1x)))")
  
  eval_f_whole_text2 = NULL
  if (n.layer.in>=2){
    for (wb.itt in 2:(n.layer.in)){
      wx.text = paste0("w", wb.itt, "x = (w", wb.itt, ")%*%sw", wb.itt-1,
                       "x + b", wb.itt)
      swx.text = paste0("sw", wb.itt, "x = as.matrix(c(relu(w", wb.itt, "x)))")
      eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
    }
  } 
  
  wb.itt.final = n.layer.in + 1
  wx.text = paste0("w", wb.itt.final, "x = (w", wb.itt.final, ")%*%sw", wb.itt.final-1,
                   "x + b", wb.itt.final)
  swx.text = paste0("sw",n.layer.in+1,"x =(w", wb.itt.final, "x)")
  eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
  
  eval_f_whole_text3 = paste0(";return(sw", n.layer.in+1, "x)}")
  
  eval_f_whole_text = paste(eval_f_whole_text1, eval_f_whole_text2,
                            eval_f_whole_text3)
  
  eval(parse(text=eval_f_whole_text))
  
  # final.pred = rep(NA, dim(data.train.in)[1])
  # for (final.pred.ind in 1:length(final.pred)){
  #   final.pred[final.pred.ind] = eval_f(as.vector(data.train.in[final.pred.ind,]))
  # }
  
  final.pred = sapply(1:(dim(data.train.in)[1]),
                      function(y){eval_f(x = as.vector(data.train.in[y,]))})
  return(final.pred)
}  

## activation function of ReLU
relu = function(x){
  return(pmax(0, x))
}

## function to fit the first DNN with ReLU as the hidden layer activation function
DNN.fit = function(data.train.scale.in, data.train.label.in, 
                   drop.rate.in, active.name.in, n.node.in, 
                   n.layer.in, max.epoch.in, validation.prop.in,
                   batch.size.in = 10^4,
                   activation.function.in = "sigmoid"){
  k_clear_session()
  build_model <- function(drop.rate.in) {
    model <- NULL
    
    model.text.1 = paste0("model <- keras_model_sequential() %>% layer_dense(units = n.node.in, activation =",
                          shQuote(active.name.in),
                          ",input_shape = dim(data.train.scale.in)[2]) %>% layer_dropout(rate=", drop.rate.in, ")%>%")
    
    model.text.2 = paste0(rep(paste0(" layer_dense(units = n.node.in, activation = ",
                                     shQuote(active.name.in),
                                     ") %>% layer_dropout(rate=", drop.rate.in, ")%>%"),
                              (n.layer.in-1)), collapse ="")
    
    ### model.text.3
    if (activation.function.in == "sigmoid"){
      model.text.3 = paste0("layer_dense(units = 1, activation = ",
                            shQuote("sigmoid"), ")")
    } else if (activation.function.in == "softmax"){
      model.text.3 = paste0("layer_dense(units = 2, activation = ",
                            shQuote("softmax"), ")")
    }
    
    # model.text.3 = paste0("layer_dense(units = 1)")
    
    eval(parse(text=paste0(model.text.1, model.text.2, model.text.3)))
    
    model %>% compile(
      # loss = MLAE,
      # loss = "mse", 
      loss = "binary_crossentropy",
      optimizer = optimizer_rmsprop(),
      # metrics = list("mean_absolute_error")
      metrics = c('accuracy')
    )
    
    model
  }
  
  out.model <- build_model(drop.rate.in)
  out.model %>% summary()
  
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 1000 == 0) cat("\n")
      cat(".")
    }
  )  
  
  history <- out.model %>% fit(
    data.train.scale.in,
    data.train.label.in,
    epochs = max.epoch.in,
    validation_split = validation.prop.in,
    verbose = 0,
    callbacks = list(print_dot_callback),
    batch_size = batch.size.in
  )
  return(list("model" = out.model, "history" = history))
}

## function to fit the second DNN with linear function as the last activation function
DNN.cutoff.fit = function(data.train.scale.in, data.train.label.in, 
                   drop.rate.in, active.name.in, n.node.in, 
                   n.layer.in, max.epoch.in, validation.prop.in,
                   batch.size.in = 10){
  k_clear_session()
  build_model <- function(drop.rate.in) {
    model <- NULL
    
    model.text.1 = paste0("model <- keras_model_sequential() %>% layer_dense(units = n.node.in, activation =",
                          shQuote(active.name.in),
                          ",input_shape = dim(data.train.scale.in)[2]) %>% layer_dropout(rate=", drop.rate.in, ")%>%")
    
    model.text.2 = paste0(rep(paste0(" layer_dense(units = n.node.in, activation = ",
                                     shQuote(active.name.in),
                                     ") %>% layer_dropout(rate=", drop.rate.in, ")%>%"),
                              (n.layer.in-1)), collapse ="")
    
    ### model.text.3
    model.text.3 = paste0("layer_dense(units = 1",
                           ")")
    # model.text.3 = paste0("layer_dense(units = 1)")
    
    eval(parse(text=paste0(model.text.1, model.text.2, model.text.3)))
    
    model %>% compile(
      # loss = MLAE,
      loss = "mean_squared_error", 
      # loss = "binary_crossentropy",
      optimizer = optimizer_rmsprop(),
      metrics = list("mean_squared_error")
      # metrics = c('accuracy')
    )
    
    model
  }
  
  out.model <- build_model(drop.rate.in)
  out.model %>% summary()
  
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 1000 == 0) cat("\n")
      cat(".")
    }
  )  
  
  history <- out.model %>% fit(
    data.train.scale.in,
    data.train.label.in,
    epochs = max.epoch.in,
    validation_split = validation.prop.in,
    verbose = 0,
    callbacks = list(print_dot_callback),
    batch_size = batch.size.in
  )
  return(list("model" = out.model, "history" = history))
}







