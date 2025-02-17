################## importing dN/dS data ##################  

mouse_rat     = read.delim("/Users/danial/Documents/postdoc_research/Evolutionary model net/V3/mouse_rat.tsv")
zebra_chicken = read.delim("/Users/danial/Documents/postdoc_research/Evolutionary model net/V3/chicken_zebra.tsv")
humandnds     = readxl::read_excel('/Users/danial/Documents/postdoc_research/Evolutionary model net/V3/humandnds.xlsx',sheet = "Sheet1")
Dros          = read.delim('/Users/danial/Documents/postdoc_research/Evolutionary model net/V3/flyDIVaS/melsubgroup_analysis_results_flydivas_v1.2.tsv')

############# null distributions #############

library(dplyr)
#human
humandnds =humandnds %>% filter(!is.na(.[[5]]))
humandnds_unsat = subset(humandnds, humandnds$dS<0.6)
#nrow(humandnds_unsat)
humandnds_random = sample_n(humandnds_unsat, 5000) %>% pull(5)

#mouse-rat
mouse_rat = mouse_rat %>% filter(!is.na(.[[ncol(mouse_rat)]]))
mouse_rat_unsat = subset(mouse_rat, mouse_rat$dS.with.Rat<0.6)
#nrow(mouse_rat_unsat)
mouse_rat_random = sample_n(mouse_rat_unsat, 5000) %>% pull(ncol(mouse_rat))

#zebra fish-chicken
zebra_chicken = zebra_chicken %>% filter(!is.na(.[[ncol(zebra_chicken)]]))
zebra_chicken_unsat = subset(zebra_chicken, zebra_chicken$dS.with.Chicken<0.6)
#nrow(zebra_chicken_unsat)
zebra_chicken_random = sample_n(zebra_chicken_unsat, 5000) %>% pull(ncol(zebra_chicken))

#Drosophila
drosdnds =Dros %>% filter(!is.na(.[[8]]))
drosdnds_unsat = subset(drosdnds, drosdnds$dS<0.6)
#nrow(drosdnds_unsat)
drosdnds_random = sample_n(drosdnds_unsat, 5000) %>% pull(8)

############# plot the null distributions ############# 

plot(density( mouse_rat_random),ylim = c(0,6),xlim = c(0,3),col ='black',main = "",ylab = "Density",xlab = "dN/dS")
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

####### plot without normalization #######
library(ggplot2)
ggplot(mydata, aes(x = position, y = W)) +
  geom_jitter(size = 1, alpha = 1, width = 0.05, height = 0) +
  
  labs(title = "", x = "", y = "") +
  
  theme_classic() +
  geom_text(aes(label = label), size = 2.) +
  theme(legend.position = "") 

###################### normalization ###################### 

mydata_normal = mydata

mydata_normal[which(grepl("\\(H-", mydata_normal$label)),]$W = mydata[which(grepl("\\(H-", mydata$label)),]$W / median(humandnds_random)
mydata_normal[which(grepl("\\(D-", mydata_normal$label)),]$W = mydata[which(grepl("\\(D-", mydata$label)),]$W / median(drosdnds_random)
mydata_normal[which(grepl("\\(M-", mydata_normal$label)),]$W = mydata[which(grepl("\\(M-", mydata$label)),]$W / median(mouse_rat_random)
mydata_normal[which(grepl("\\(Z-", mydata_normal$label)),]$W = mydata[which(grepl("\\(Z-", mydata$label)),]$W / median(zebra_chicken_random)

####### plot with normalization #######

ggplot(mydata_normal, aes(x = position, y = W)) +
  geom_jitter(size = 1, alpha = 1, width = 0.05, height = 0) +
  
  labs(title = "", x = "", y = "") +
  
  theme_classic() +
  geom_text(aes(label = label), size = 2.) +
  theme(legend.position = "") 




############## test variance before normalization ##############


#Simple model with no variance structure and gene as random effect

model1= lme(W ~ position, random = ~1 | label, data = mydata)

#This model allows variance to vary across the two groups (i.e., upstream vs downstream) and gene as random effect

model2 = lme(W ~ position, random = ~1 | label, 
             weights = varIdent(form = ~1 | position), data = mydata)


#summary of the better model
summary(model2)

#model comparison
anova(model1, model2)




############## test variance after normalization ##############

#Estimating the two variance
round(var(subset(mydata_normal, mydata_normal$position == "Upstream")$W),4)
round(var(subset(mydata_normal, mydata_normal$position == "Downstream")$W),4)
fligner.test(list(subset(mydata_normal, mydata_normal$position == "Upstream")$W, subset(mydata_normal, mydata_normal$position == "Downstream")$W))
library(nlme)

#Simple model with no variance structure and gene as random effect

model1= lme(W ~ position, random = ~1 | label, data = mydata_normal)

#This model allows variance to vary across the two groups (i.e., upstream vs downstream) and gene as random effect

model2 = lme(W ~ position, random = ~1 | label, 
             weights = varIdent(form = ~1 | position), data = mydata_normal)


#summary of the better model
summary(model2)

#model comparison
anova(model1, model2)


############ End of the code ############