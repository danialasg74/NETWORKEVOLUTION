################## importing dN/dS data ##################  

mouse_rat     = read.delim(".../mouse_rat.tsv")
zebra_chicken = read.delim(".../chicken_zebra.tsv")
humandnds     = readxl::read_excel('.../humandnds.xlsx',sheet = "Sheet1")
Dros          = read.delim('.../melsubgroup_analysis_results_flydivas_v1.2.tsv')

############# null distributions #############

library(dplyr)
#human
humandnds =humandnds %>% filter(!is.na(.[[5]]))
humandnds_unsat = subset(humandnds, humandnds$dS<0.6)
humandnds_random = sample_n(humandnds_unsat, 5000) %>% pull(5)

#mouse-rat
mouse_rat$W = mouse_rat$dN.with.Rat/ mouse_rat$dS.with.Rat
mouse_rat = mouse_rat %>% filter(!is.na(.[[ncol(mouse_rat)]]))
mouse_rat_unsat = subset(mouse_rat, mouse_rat$dS.with.Rat<0.6)
mouse_rat_random = sample_n(mouse_rat_unsat, 5000) %>% pull(ncol(mouse_rat))

#zebra fish-chicken
zebra_chicken$W = zebra_chicken$dN.with.Chicken/zebra_chicken$dS.with.Chicken
zebra_chicken = zebra_chicken %>% filter(!is.na(.[[ncol(zebra_chicken)]]))
zebra_chicken_unsat = subset(zebra_chicken, zebra_chicken$dS.with.Chicken<0.6)
zebra_chicken_random = sample_n(zebra_chicken_unsat, 5000) %>% pull(ncol(zebra_chicken))

#Drosophila
drosdnds =Dros %>% filter(!is.na(.[[8]]))
drosdnds_unsat = subset(drosdnds, drosdnds$dS<0.6)
drosdnds_random = sample_n(drosdnds_unsat, 5000) %>% pull(8)

############# plot the null distributions ############# 

plot(density( mouse_rat_random),ylim = c(0,10),xlim = c(0,3),col ='black',main = "",ylab = "Density",xlab = "dN/dS")
lines(density(zebra_chicken_random),col ='green')
lines(density(drosdnds_random),ylim = c(0,10),xlim = c(0,5),col ='brown')
lines(density(humandnds_random),ylim = c(0,10),xlim = c(0,5),col ='purple')

legend("topright", legend = c("Mouse-Rat", "Zebra-Chicken", "Drosophila",'human'), 
       col = c("black", "green", "brown", 'purple'), lty = 1, cex = 0.8)

##############################  dN/dS of NFLs ########################################

################ Mouse-rat ################

#Mouse upstream NFL

#### Labels: Dkk3;A20;Socs1;TRIM38, SIGIRR, ptch1, HHIP, lefty
mouse_up_df = mouse_rat[mouse_rat$Gene.stable.ID %in% c("ENSMUSG00000030772", "ENSMUSG00000019850", "ENSMUSG00000038037", "ENSMUSG00000064140", "ENSMUSG00000025494" , "ENSMUSG00000021466" , "ENSMUSG00000064325", "ENSMUSG00000038793"),]
mouse_up_df_uniq = subset(mouse_up_df, !duplicated(mouse_up_df$Gene.stable.ID))
mouse_up_df_uniq$names = c("Lefty", "SIGIRR", "HHIP","ptch1", "SOCS1", "A20", "Dkk3")
mouse_up_df_uniq$dnds = mouse_up_df_uniq$dN.with.Rat / mouse_up_df_uniq$dS.with.Rat

#Mouse downstream NFL
#### Labels: Axin2; LATS1; LATS2, Hes1, CITED2
mouse_do_df = mouse_rat[mouse_rat$Gene.stable.ID %in% c("ENSMUSG00000000142", "ENSMUSG00000040021", "ENSMUSG00000021959","ENSMUSG00000022528", "ENSMUSG00000039910"),]
mouse_do_df_uniq = subset(mouse_do_df, !duplicated(mouse_do_df$Gene.stable.ID))
mouse_do_df_uniq$name = c('Axin2', 'LATS2',"CITED2",'LATS1', "Hes1")
mouse_do_df_uniq$dnds = mouse_do_df_uniq$dN.with.Rat / mouse_do_df_uniq$dS.with.Rat

################ Zebra fish-chicken ################

#Zebra upstream NFL
#### Labels: Dkk3; A20; SOCS1,,SIGIRR, ptch1, HHIP, lefty
zebra_up_df = zebra_chicken[zebra_chicken$Gene.stable.ID %in% c("ENSTGUG00000016109", "ENSTGUG00000010895", "ENSTGUG00000004867","ENSTGUG00000006773", "ENSTGUG00000000532", "ENSTGUG00000002714", "ENSTGUG00000004196"),]
zebra_up_df_uniq = subset(zebra_up_df, !duplicated(zebra_up_df$Gene.stable.ID))
zebra_up_df_uniq$name  = c("A20","Dkk3", "SOCS1","SIGIRR", "Lefty", "HHIP", "ptch1")
zebra_up_df_uniq$dnds = zebra_up_df_uniq$dN.with.Chicken / zebra_up_df_uniq$dS.with.Chicken


#Zebra downstream NFL
#### Labels: AXIN2; LATS1; LATS2,Hes1, CITED2
zebra_do_df = zebra_chicken[zebra_chicken$Gene.stable.ID %in% c("ENSTGUG00000004202", "ENSTGUG00000011347", "ENSTGUG00000011364","ENSTGUG00000009134", "ENSTGUG00000027340"),]
zebra_do_df_uniq = subset(zebra_do_df, !duplicated(zebra_do_df$Gene.stable.ID))
zebra_do_df_uniq$name = c('LATS2','CITED2', "LATS1", "HES1", "AXIN2")
zebra_do_df_uniq$dnds = zebra_do_df_uniq$dN.with.Chicken / zebra_do_df_uniq$dS.with.Chicken

################ Drosophila ################

#Dros upstream NFL
Dros_up_df = Dros[Dros$id %in% c("FBgn0034647", "FBgn0037906", "FBgn0043575"),]

#Dros downstream NFL
Dros_do_df = Dros[Dros$id %in% c("FBgn0016917", "FBgn0000250", "FBgn0038134"),]

################ human ################ 

#human upstream NFL

hum_up_df = humandnds[humandnds$`Ensembl Gene ID` %in% c("ENSG00000185338", "ENSG00000118503", "ENSG00000050165", "ENSG00000112343", "ENSG00000185187", "ENSG00000185920", "ENSG00000164161", "ENSG00000143768"),]
hum_do_df = humandnds[humandnds$`Ensembl Gene ID` %in% c("ENSG00000100906", "ENSG00000139318", "ENSG00000168646", "ENSG00000131023", "ENSG00000150457", "ENSG00000114315", "ENSG00000164442"),]

################################ combine dN/dS of NFLs for all species ################################

####### just upstream combined ####### 

Upstream_NFL = data.frame(
  
  W = c(mouse_up_df_uniq$dnds,
        zebra_up_df_uniq$dnds,
        Dros_up_df$omega,
        hum_up_df$`dN/dS`),
  
  position = "Upstream",
  
  label= c(
    "LEFTY (M-Nodal)",
    "SIGIRR (M-NFkB)",
    "HHIP (M-Shh)",
    "PTCH1 (M-Shh)",
    "SOCS1 (M-JAK)",
    "A20 (M-NFkB)",
    "DKK3 (M-Wnt)",
    
    
    "A20 (Z-NFkB)",
    "DKK3 (Z-Wnt)",
    "SOCS1 (Z-JAK)",
    "SIGIRR (Z-NFkB)",
    "LEFTY (Z-Nodal)",
    "HHIP (Z-Shh)",
    "PTCH1 (Z-Shh)",
    
    
    "Pirk (D-Imd)",
    "PGRP-SC2 (D-Imd)",
    "PGRP-LB (D-Imd)",
    
    
    "SOCS1 (H-JAK)" ,
    "A20 (H-NFkB)",
    "LEFTY (H-Nodal)",
    "DKK3 (H-Wnt)"  ,
    "SIGIRR (H-NFkB)",
    "PTCH1 (H-Shh)" ,
    "TRIM38 (H-NFkB)",
    "HHIP (H-Shh)")
)

####### just downstream combined ####### 

downstream_NFL = data.frame(
  
  W = c(mouse_do_df_uniq$dnds,
        zebra_do_df_uniq$dnds,
        Dros_do_df$omega,
        hum_do_df$`dN/dS`),
  
  position = "Downstream",
  
  
  label= c(
    "AXIN2 (M-Wnt)",
    "LATS2 (M-Hippo)",
    "CITED2 (M-NFkB)",
    "LATS1 (M-Hippo)",
    "HES1 (M-Notch)",
    
    
    "LATS2 (Z-Hippo)",
    "CITED2 (Z-NFkB)",
    "LATS1 (Z-Hippo)",
    "HES1 (Z-Notch)",
    "AXIN2 (Z-Wnt)",
    
    
    "Stat92E (D-Imd)",
    "WntD (D-Toll)",
    "Cact (D-Toll)",
    
    
    "CITED2 (H-NFkB)",      
    "LATS2 (H-Hippo)",       
    "DUSP6 (H-MAPK)" ,      
    "HES1 (H-Notch)" ,       
    "IKBa (H-NFkB)"  ,    
    "AXIN2 (H-Wnt)"  ,     
    "LATS1 (H-Hippo)"  )
)

################ comine all ################

mydata = rbind(Upstream_NFL, downstream_NFL)
mydata$position = factor(mydata$position , levels = c('Upstream' , 'Downstream'))

####### plot raw data #######
library(ggplot2)
ggplot(mydata, aes(x = position, y = W)) +
  geom_jitter(size = 1, alpha = 1, width = 0.05, height = 0) +
  
  labs(title = "", x = "", y = "") +
  
  theme_classic() +
  geom_text(aes(label = label), size = 1.) +
  theme(legend.position = "") +scale_x_discrete(expand = expansion(mult = c(0.3, 2)))


######################## bootstraping ########################

#combine all null distributions
nulldnds  = c(mouse_rat_random,zebra_chicken_random,drosdnds_random,humandnds_random)  

mypval        = c()
fixed_down    = c()
fixed_up      = c()
var_down      = c()
var_up        = c()
upstream_pval = c()

for (i in 1:10000) {
  repeat {  # Loop until both models converge
    
    # Sample null distribution
    SampleNull = sample(nulldnds,20)
    # Unique labels for backgrounds
    labelsnull = as.character(c(1:20))
    # Dataframe of null values
    null_df = data.frame(W = SampleNull, position = "background", label = labelsnull)
    # Update dn/ds with new background dn/ds
    mydata_bootstap = rbind(mydata, null_df)
    # Levels of positions
    mydata_bootstap$position = factor(mydata_bootstap$position, levels = c('Downstream', "Upstream", "background"))
    
    # Try fitting model1 first
    model1_result = tryCatch({
      model1 = lme(W ~ position, random = ~1 | label, data = mydata_bootstap)
      list(success = TRUE, model1 = model1)
    }, error = function(e) {
      list(success = FALSE, model1 = NULL)  # If error, indicate failure
    })
    
    # If model1 fails, resample and retry
    if (!model1_result$success) {
      next  # Skip to next iteration (repeat loop)
    }
    
    model1 = model1_result$model1  # Extract successful model1
    
    # Try fitting model2
    model2_result = tryCatch({
      model2 = lme(W ~ position, random = ~1 | label,
                   weights = varIdent(form = ~1 | position), data = mydata_bootstap)
      list(success = TRUE, model2 = model2)
    }, error = function(e) {
      list(success = FALSE, model2 = NULL)  # If error, indicate failure
    })
    
    # If model2 fails, resample and retry
    if (!model2_result$success) {
      next  # Skip to next iteration (repeat loop)
    }
    
    model2 = model2_result$model2  # Extract successful model2
    break  # Exit repeat loop if both models converge
  }
  
  # Model comparison
  modelcompare = anova(model1, model2)
  
  #Model summary
  summ1 = summary(model1)
  summ2 = summary(model2)
  
  # p-value extraction
  pval = as.numeric(modelcompare$`p-value`)[2]
  mypval = c(mypval, pval)
  
  if (pval<0.05){
    #upstream effect pval
    upstream_pval = c(upstream_pval, summ2[["tTable"]]["positionUpstream", "p-value"])
    #Variance of downstream (model2)
    var_down = c(var_down, exp(coef(model2$modelStruct$varStruct))[2])
    #Variance of upstream (model2)
    var_up = c(var_up, exp(coef(model2$modelStruct$varStruct))[1])
    #fixed effects
    fixed_up = c(fixed_up, model2$coefficients$fixed[2])
    fixed_down = c(fixed_down, model2$coefficients$fixed[1])
  
  }
  else{
    #upstream effect pval
    upstream_pval = c(upstream_pval, summ1[["tTable"]]["positionUpstream", "p-value"])
    #fixed effects
    fixed_up = c(fixed_up, model1$coefficients$fixed[2])
    fixed_down = c(fixed_down, model1$coefficients$fixed[1])

  }
}



#pval distribution
hist(mypval, breaks=100 , xlab = "p-val for model comparison", main = "", cex.main = 0.8, 
     cex.lab = 1,   
     cex.axis = 0.7)
abline(v = 0.05, col = "red", lty = 2, lwd = 2)
# Add the result as text to the plot
prop_significant = length(which(mypval<0.05))/length(mypval)
text(x = 0.5, y = max(hist(upstream_pval, plot = FALSE)$counts) * 0.4, 
     labels = paste("Proportion < 0.05: ", round(prop_significant, 3)), 
     col = "red", cex = 1, font = 1)

#0.1 cutoff
length(which(mypval<0.1))/length(mypval)

#plot var
hist1_var = hist(var_down, breaks=100, plot=FALSE)
hist2_var = hist(var_up, breaks=100, plot=FALSE)
plot(hist1_var, col=rgb(1, 0, 0, 0.5), main = "Variance",xlab="relative to background")
plot(hist2_var, col=rgb(0, 0, 1, 0.5), add=TRUE)
legend("topright", legend=c("Downstream", "Upstream"), fill=c(rgb(1, 0, 0, 0.5), rgb(0, 0, 1, 0.5)))



# Plot the histogram
hist(upstream_pval, breaks=100, xlab = "p-val (fixed effect of upstream)", main = "")

# Compute the proportion of p-values < 0.05
prop_significant <- length(which(upstream_pval < 0.05)) / length(upstream_pval)
# Add the result as text to the plot
text(x = 0.5, y = max(hist(upstream_pval, plot = FALSE)$counts) * 0.4, 
     labels = paste("Proportion < 0.05: ", round(prop_significant, 3)), 
     col = "red", cex = 1, font = 1)
abline(v = 0.05, col = "red", lty = 2, lwd = 2)
#0.1 cutoff
length(which(upstream_pval<0.1))/length(upstream_pval)






#fixed effect
hist1_fixed = hist(fixed_down, breaks=10, plot=FALSE)
hist2_fixed = hist(fixed_up, breaks=20, plot=FALSE)
plot(hist1_fixed, col=rgb(1, 0, 0, 0.5), xlim = c(0.06,0.085), main = "", xlab="Fixed Effect")
plot(hist2_fixed, col=rgb(0, 0, 1, 0.5), add=TRUE)
legend("topright", legend=c("Downstream", "Upstream"), fill=c(rgb(1, 0, 0, 0.5), rgb(0, 0, 1, 0.5)))

