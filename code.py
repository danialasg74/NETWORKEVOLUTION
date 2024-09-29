import numpy as np
import random
from numba import jit
#Functions and initial values
#############################

#Function for calculating hamming distance
def hamming_distance(s1, s2):
    distance = sum(el1 != el2 for el1, el2 in zip(s1, s2))
    return distance
#Function for calculating hamming distance between I and P (always inhibitory)
def hamming_distance2(s1, s2):
    distance = sum(el1 != el2 for el1, el2 in zip(s1, s2))
    if distance > 5:
        distance = 10 - distance
    return distance
    
#System of ODE (Euler)
@jit(nopython=False)
def funcs(NUMBERINFECTED,time_length, YINF,MUIND,phiB,phiO):
    for ind in range(NUMBERINFECTED):
        h = 1/np.sum(np.abs(MUIND[ind]))
        for t in range(time_length-1):
        	#Pathogen
            YINF[ind][0][t+1] = YINF[ind][0][t] + h*( (-phiB*YINF[ind][0][t] + -MUIND[ind][0][3] * YINF[ind][3][t] * (MUIND[ind][0][3] < 0 and (1.0 - YINF[ind][0][t]) or YINF[ind][0][t])) + (PROLIF * YINF[ind][0][t] * (1 - YINF[ind][0][t]) )  )
            #Receptor
            YINF[ind][1][t+1] = YINF[ind][1][t] + h*( (-phiO*YINF[ind][1][t] +  MUIND[ind][1][0] * YINF[ind][0][t] * (MUIND[ind][1][0] > 0 and (1.0 - YINF[ind][1][t]) or YINF[ind][1][t]) ) +(MUIND[ind][1][4] * YINF[ind][4][t] * (MUIND[ind][1][4] > 0 and (1.0 - YINF[ind][1][t]) or YINF[ind][1][t]) ))
            #Activator
            YINF[ind][2][t+1] = YINF[ind][2][t] + h*(  -phiO*YINF[ind][2][t] +  MUIND[ind][2][1] * YINF[ind][1][t] * (MUIND[ind][2][1] > 0 and (1.0 - YINF[ind][2][t]) or YINF[ind][2][t]))
            #Immunity
            YINF[ind][3][t+1] = YINF[ind][3][t] + h*( (-phiO*YINF[ind][3][t] +  MUIND[ind][3][2] * YINF[ind][2][t] * (MUIND[ind][3][2] > 0 and (1.0 - YINF[ind][3][t]) or YINF[ind][3][t]) ) +(MUIND[ind][3][5] * YINF[ind][5][t] * (MUIND[ind][3][5] > 0 and (1.0 - YINF[ind][3][t]) or YINF[ind][3][t])) )
            #Upstream regulator
            YINF[ind][4][t+1] = YINF[ind][4][t] + h*(  -phiO*YINF[ind][4][t] +  MUIND[ind][4][2] * YINF[ind][2][t] * (MUIND[ind][4][2] > 0 and (1.0 - YINF[ind][4][t]) or YINF[ind][4][t]))
            #Downstream regulator
            YINF[ind][5][t+1] = YINF[ind][5][t] + h*(  -phiO*YINF[ind][5][t] +  MUIND[ind][5][2] * YINF[ind][2][t] * (MUIND[ind][5][2] > 0 and (1.0 - YINF[ind][5][t]) or YINF[ind][5][t]))

#Vector of interaction before infection
#Protein i is affected by j (e.g., i = 0 (receptor) is affected by j = 3 (Up NFL))
cmbefore = [ 
    #y0  y1  y2  y3  y4
    
    [0,  0,  0,  1,  0], #y0 [Receptor]
    
    [1,  0,  0,  0,  0], #y1 [Activator]
    
    [0,  1,  0,  0,  1], #y2 [Immunity]
    
    [0,  1,  0,  0,  0], #y3 [Up NFL]
    
    [0,  1,  0,  0,  0], #y4 [Down NFL]
]

#Vector of interaction after infection
#Protein i is affected by j (e.g., i = 0 (Pathogen) is affected by j = 3 (immunity))
cm = [
    #y0  y1  y2  y3  y4  y5
    [0,  0,  0,  1,  0,  0], #y0 [Pathogen]
    
    [1,  0,  0,  0,  1,  0], #y1 [Receptor]
    
    [0,  1,  0,  0,  0,  0], #y2 [Activator]
    
    [0,  0,  1,  0,  0,  1], #y3 [Immunity]
    
    [0,  0,  1,  0,  0,  0], #y4 [Up NFL]
    
    [0,  0,  1,  0,  0,  0], #y5 [Down NFL]
]

#Parameters
#Bacterial proliferation rate
PROLIF =0.1
#Size of the proteins
DOMAINSIZE = 10
#Number of domains
NDOMAINS = 3
#Number of host proteins
NUMPRO = 5
#Number of pathogens
NUMPAR = 2000
#Number of hosts
NUMIND = 2000
#Pathogen degradation rate 
phiB = 0
#Host protein degradation rate 
phiO = 0.15
#relative effect for the cost of infection
alpha = 2
#relative effect for the cost of the immune response
beta = 1
#Host life span
time_length = 1000
#The rate of infection
prob_encounter= 1
#Mutation rate of the individuals
MUTATIONRATEI = 0.001
#Mutation rate of the pathogens
MUTATIONRATEP = MUTATIONRATEI*2
#Number of generations
NUMBERGENR = 20000

#Focal and temporal consensus sequence
FOCALINPU = 0
FOCALOUPU = 0
FOCALNEUT = 0
TEMPOINPU = 0
TEMPOOUPU = 0
TEMPONEUT = 0

#Rate of evolution
RATEINPUT = []
RATEOUPUT = []
KS_VALUES = []

#Counter for generation
CURRENTGEN = 0

#Vector of hamming distance in time for all individuals
HAM_TIME = []
HAM_INF = []
#Initialize indiviudals with random proteins
PARENTS   = [ [[[random.randint(0, 1) for _ in range(DOMAINSIZE)] for _ in range(NDOMAINS)] for _ in range(NUMPRO)] for _ in range(NUMIND) ]

#Initialize pathogen with random proteins 
PARASITE  = [ [[[random.randint(0, 1) for _ in range(DOMAINSIZE)] for _ in range(NDOMAINS)] for _ in range(1)] for _ in range(NUMPAR) ]

#Average fitness
AVERAGEFH = []
AVERAGEFP = []

#Fitness variance
VARFITH = []
VARFITP = []

#Evolution
##############

for _ in range(NUMBERGENR):
    
    #At every 5th generation calculate k
    if CURRENTGEN % 5 == 0:

        #Making consensus sequences
        CONSENSUSINPU = []
        CONSENSUSOUPU = []
        CONSENSUSNEUT = []

        #for each protein
        for prot in range(NUMPRO):
            
            #For every locus 
            sum_inpu = []
            sum_oupu = []
            sum_neut = []
            
            for locus in range(DOMAINSIZE):
                
                #For the 3 domains
                sumloc_inpu = 0
                sumloc_oupu = 0
                sumloc_neut = 0
    
                #sum 1s across the population
                for ind in range(NUMIND):   
                    
                    #Across individuals
                    sumloc_inpu += PARENTS[ind][prot][0][locus]
                    sumloc_oupu += PARENTS[ind][prot][1][locus]
                    sumloc_neut += PARENTS[ind][prot][2][locus]
                
                #Across loci
                sum_inpu.append(sumloc_inpu/NUMIND)
                sum_oupu.append(sumloc_oupu/NUMIND)
                sum_neut.append(sumloc_neut/NUMIND)

            #Across proteins    
            CONSENSUSINPU.append(sum_inpu)
            CONSENSUSOUPU.append(sum_oupu)
            CONSENSUSNEUT.append(sum_neut)
        
        #if t ==0 calculate the focal
        if CURRENTGEN == 0:
            
            FOCALINPU = CONSENSUSINPU
            FOCALOUPU = CONSENSUSOUPU
            FOCALNEUT = CONSENSUSNEUT
            
        #Otherwise calculate the temporal
        else:
            
            TEMPOINPU = CONSENSUSINPU
            TEMPOOUPU = CONSENSUSOUPU
            TEMPONEUT = CONSENSUSNEUT
            
        #If both focal and temporal exist (if t != 0)
        if CURRENTGEN > 0:
            
            #Calculate K for every protein
            ALLPROTKS_check = []
            ALLPROTKS       = []
            ALLPROTKa_I     = []
            ALLPROTKa_O     = []
            
            #For every protein calculate the ks
            for prot in range(NUMPRO):
                
                ks   = 0

                
                #check for saturation
                for bit in range(DOMAINSIZE):
                    
                    ks     += FOCALNEUT[prot][bit] + TEMPONEUT[prot][bit] - 2*(FOCALNEUT[prot][bit]*TEMPONEUT[prot][bit])
                
                #k for every protein
                ALLPROTKS.append(ks)
                
                #for each protein, if ks > 0.5, set focal to temporal
                if ALLPROTKS[prot] > 0.5:
                    
                    FOCALNEUT[prot] = TEMPONEUT[prot]
                    FOCALINPU[prot] = TEMPOINPU[prot]
                    FOCALOUPU[prot] = TEMPOOUPU[prot]
                    ALLPROTKa_I.append(0)
                    ALLPROTKa_O.append(0)

                else:
                    #calculate k
                    ka_i = 0
                    ka_o = 0
                    
                    for bit in range(DOMAINSIZE):
                        
                        ka_i   += FOCALINPU[prot][bit] + TEMPOINPU[prot][bit] - 2*(FOCALINPU[prot][bit]*TEMPOINPU[prot][bit])
                        ka_o   += FOCALOUPU[prot][bit] + TEMPOOUPU[prot][bit] - 2*(FOCALOUPU[prot][bit]*TEMPOOUPU[prot][bit])
                       
                    ALLPROTKa_I.append(ka_i)
                    ALLPROTKa_O.append(ka_o)

            #Rate of evolution at each interval
            KS_VALUES.append(np.array(ALLPROTKS))

            RATEINPUT.append(np.array(ALLPROTKa_I)/np.array(ALLPROTKS))
            RATEOUPUT.append(np.array(ALLPROTKa_O)/np.array(ALLPROTKS))

    #Random encounters
    indices = list(range(NUMIND))
    random.shuffle(indices)

    #System of host parasite
    SYSTEMSHP  = []

    #Encountered parasites with hosts
    ENCOUNTERP = []
    #Encountered hosts with parasites
    ENCOUNTERH = []
    
    #Whcih hosts and parasites interact
    for ind in range(NUMIND):
        
        if random.random() < prob_encounter:
            
            #pick a random parasite
            randomparasite = indices[ind]
            
            #The parasite that encounters a host
            ENCOUNTERP.append(randomparasite)
            
            #The host that encounters parasite
            ENCOUNTERH.append(ind)
            
            #Form a host-parasite system
            SYSTEMSHP.append(PARASITE[randomparasite] + PARENTS[ind])
    
    #Index values for those without encounter
    NOTENCOUNTERH  = [i for i in list(range(NUMIND)) if i not in ENCOUNTERH]
    NOTENCOUNTERP  = [i for i in indices if i not in ENCOUNTERP]
    
    #Vector to store hamming distance prior infection
    HAM_ALL = [[[0 for _ in range(NUMPRO)] for _ in range(NUMPRO)] for _ in range(NUMIND)]
    
    #Vector to store hamming distance after infection
    HAM_IND = [[[0 for _ in range(NUMPRO+1)] for _ in range(NUMPRO+1)] for _ in range(len(SYSTEMSHP))]
    
    #Calculate Hamming distances for every individuals before infection
    for i in range(len(cmbefore)):
        for j in range(len(cmbefore[0])):
            #If proteins have a connection within the network
            if cmbefore[i][j] ==1:  
                for ind in range(NUMIND):              #Input Domain of i =0   output Domain of j = 1
                    HAM_ALL[ind][i][j]  = hamming_distance(PARENTS[ind][i][0], PARENTS[ind][j][1])
    
    #Store Hamming distance matrix over generations
    HAM_TIME.append(HAM_ALL)
    
    #Calculate Hamming distances for infected individuals
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            #I always inhibits P (or has no effect)
            if i == 0 and j == 3:
                for ind in range(len(SYSTEMSHP)):
                    HAM_IND[ind][i][j]  = hamming_distance2(SYSTEMSHP[ind][i][0], SYSTEMSHP[ind][j][1])
            elif cm[i][j] == 1: 
                for ind in range(len(SYSTEMSHP)):            #Input Domain of i  output Domain of j
                    HAM_IND[ind][i][j]  = hamming_distance(SYSTEMSHP[ind][i][0], SYSTEMSHP[ind][j][1])
    
    HAM_INF.append(HAM_IND)
    #Calculate coefficients for solving ODEs
    MUIND = 1- np.multiply(2,HAM_IND)/DOMAINSIZE
    
    NUMBERINFECTED = len(SYSTEMSHP)
    YINF = [np.zeros((len(cm), time_length)) for _ in range(NUMBERINFECTED)]

    #Initial condition
    for ind in range(NUMBERINFECTED):
        YINF[ind][0][0] = 1

    YINF = np.array(YINF)
    
    #integrate
    funcs(NUMBERINFECTED,time_length, YINF,MUIND,phiB,phiO)
    
    #Calculate host fitness upon encounter
    FITNESSINFH = np.zeros(len(SYSTEMSHP))
    for ind in range(len(SYSTEMSHP)):
        FITNESSINFH[ind] = np.exp(-(BEFFECT*np.average(YINF[ind][0]) + IEFFECT*np.average(YINF[ind][3])) )
        
    #Calculate parasite fitness upon encounter
    FITNESSINFP = np.zeros(len(SYSTEMSHP))  
    for ind in range(len(SYSTEMSHP)):
        FITNESSINFP[ind] = np.average(YINF[ind][0])
        
    FitnessH = np.zeros(NUMIND)
    #Hosts' fitness ==1 when not encountering pathogens
    FitnessH[NOTENCOUNTERH] = 1
    
    #Hosts' fitness when encountering pathogens
    for i in range(len(FITNESSINFH)):
        FitnessH[ENCOUNTERH[i]] = FITNESSINFH[i]
    
    #Parasite fitness ==0 when not infecting  
    FitnessP = np.zeros(NUMIND)
    FitnessP[NOTENCOUNTERP] = 0
    
    #Parasite fitness when infecting
    for i in range(len(FITNESSINFP)):
        FitnessP[ENCOUNTERP[i]] = FITNESSINFP[i]
        
    #Avergae fitness across generations
    AVERAGEFH.append(np.average(FitnessH))
    AVERAGEFP.append(np.average(FitnessP))
    
    #Fitness variance
    VARFITH.append(np.var(FitnessH))
    VARFITP.append(np.var(FitnessP))
    
    #Reproduction parameters
    NEWGENERATION = []
    NUMBEROFFSPRING = NUMIND
    
    #Reproduction simulation
    while NUMBEROFFSPRING:
        
        #pick Parent 1 based on its fitness
        i = random.choices(range(NUMIND), weights=FitnessH)[0]
        j = i
        while j ==i:
            #Parent 2 (when i !=j) based on its fitness
            j = random.choices(range(NUMIND), weights=FitnessH)[0]
            
        #Offspring of parent i and j
        OFFSPRING = []
    
        #Make the new offspring by combining proteins from parents i and j
        for prot in range(NUMPRO):
        
            #For each protein recombination happens at:
            #Recombination domain:
            Recomdom = random.randint(0, NDOMAINS-1)
            #Recombanation site:
            Recomsit = random.randint(0, DOMAINSIZE-1)
            
            #A new protein that is made by domains that either undergo recom or don't
            NEWPRO = []
    
            for dom in range(NDOMAINS):
                
                NEWDOMAIN = []
                
                if dom == Recomdom:
                    
                    #Recombination
                    NEWDOMAIN += [PARENTS[i][prot][dom][:Recomsit] + PARENTS[j][prot][dom][Recomsit:DOMAINSIZE]]
                else:
                    
                    #No recombination
                    if random.random() < 0.5:
                        NEWDOMAIN += [PARENTS[i][prot][dom]]
                    else:
                        NEWDOMAIN += [PARENTS[j][prot][dom]]
             
                #Make a new protein from new domains
                NEWPRO += NEWDOMAIN
                
            #Make a new offspring from new proteins
            OFFSPRING.append(NEWPRO)
             
        #Mutate the new offspring with prob = MUTATIONRATEI
        
        #Pick a random protein to mutate
        randomprotein = random.randint(0, 4)
        
        if random.random() < MUTATIONRATEI/3:
            #mutation in input domain
            randomlnucleo = random.randint(0, DOMAINSIZE-1)
            #Mutate
            OFFSPRING[randomprotein][0][randomlnucleo] ^= 1
            
         #Mutation at input domain with probability MUTATIONRATEI/3
        if random.random() < MUTATIONRATEI/3:
            #mutation in output domain
            randomlnucleo = random.randint(0, DOMAINSIZE-1)
            OFFSPRING[randomprotein][1][randomlnucleo] ^= 1 
        
        if random.random() < MUTATIONRATEI/3:
            #mutation in neutral domain
            randomlnucleo = random.randint(0, DOMAINSIZE-1)
            OFFSPRING[randomprotein][2][randomlnucleo] ^= 1   
            
        #Add the new offspring to the new generation
        NEWGENERATION.append(OFFSPRING)
        
        #counter
        NUMBEROFFSPRING = NUMBEROFFSPRING -1
        
    NUMBEROFFSPARASITE = NUMPAR
    PARASITENEWGEN = []
    
    while NUMBEROFFSPARASITE: 
        
        #Select a parasite based on fitness values
        PNEWINDEX = random.choices(range(NUMPAR), weights=FitnessP)[0]
        SELECTEDP = PARASITE[PNEWINDEX]
        
        #Parasite only has 1 protein
        proteinnindx = 0
        
        #Mutate the parasite
        if random.random() < MUTATIONRATEP/3:
            #mutation in input domain
            randomlnucleo = random.randint(0, DOMAINSIZE-1)
            SELECTEDP[proteinnindx][0][randomlnucleo] ^= 1
        
        if random.random() < MUTATIONRATEP/3:
            #mutation in output domain
            randomlnucleo = random.randint(0, DOMAINSIZE-1)
            SELECTEDP[proteinnindx][1][randomlnucleo] ^= 1 
        
        if random.random() < MUTATIONRATEP/3:
            #mutation in neutral domain
            randomlnucleo = random.randint(0, DOMAINSIZE-1)
            SELECTEDP[proteinnindx][2][randomlnucleo] ^= 1 
            
            
        PARASITENEWGEN.append(SELECTEDP)
        
        #counter
        NUMBEROFFSPARASITE = NUMBEROFFSPARASITE - 1
    
    #current Generation
    CURRENTGEN =  CURRENTGEN + 1
    print(CURRENTGEN, flush = True)
    #Set the new generation as the current generation
    PARENTS  = NEWGENERATION
    PARASITE = PARASITENEWGEN
    
    
#Effect of I on B
i = 0
j = 3
IONB    = []
IONBVAR = []

#At every time point
for t in range(CURRENTGEN):
    myvec1 = 0
    myvec2 = []
    mylength = len(HAM_INF[t])
    #Get the avergae mu for UPNFL (i = 3; j = 1)
    for ind in range(mylength):
        myvec1 += (1- np.multiply(2,HAM_INF[t][ind][i][j])/DOMAINSIZE)
        myvec2.append((1- np.multiply(2,HAM_INF[t][ind][i][j])/DOMAINSIZE))
    IONBVAR.append(np.var(myvec2))
    IONB.append(myvec1/NUMIND)


#Effect of U on R    
i = 1-1
j = 4-1
UPONRECEPT   = []
UPONRECEPTVAR = []

#At every time point
for t in range(CURRENTGEN):
    myvec1 = 0
    myvec2 = []
    #Get the avergae mu for UPNFL (i = 3; j = 1)
    for ind in range(2000):
        myvec1 += (1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE)
        myvec2.append((1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE))
    UPONRECEPTVAR.append(np.var(myvec2))
    UPONRECEPT.append(myvec1/NUMIND)
    
    

#Effect of D on I 
i = 3-1
j = 5-1
DOWNONRECEPT   = []
DOWNONRECEPTVAR = []

#At every time point
for t in range(CURRENTGEN):
    myvec1 = 0
    myvec2 = []
    #Get the avergae mu for DOWNNFL (i = 3; j = 1)
    for ind in range(2000):
        myvec1 += (1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE)
        myvec2.append((1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE))
    DOWNONRECEPTVAR.append(np.var(myvec2))
    DOWNONRECEPT.append(myvec1/NUMIND)


#Effect of A on U   
i = 4-1
j = 2-1
UPACT   = []
UPACTVAR = []

#At every time point
for t in range(CURRENTGEN):
    myvec1 = 0
    myvec2 = []
    #Get the avergae mu for UPNFL (i = 3; j = 1)
    for ind in range(2000):
        myvec1 += (1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE)
        myvec2.append((1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE))
    UPACTVAR.append(np.var(myvec2))
    UPACT.append(myvec1/NUMIND)
 
 

#Effect of A on D   
i = 5-1
j = 2-1
DOWNACT   = []
DOWNACTVAR = []

#At every time point
for t in range(CURRENTGEN):
    myvec1 = 0
    myvec2 = []
    #Get the avergae mu for DOWNNFL (i = 3; j = 1)
    for ind in range(2000):
        myvec1 += (1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE)
        myvec2.append((1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE))
    DOWNACTVAR.append(np.var(myvec2))
    DOWNACT.append(myvec1/NUMIND)
    
    
    
    
    
#Effect of P on R
i = 1
j = 0
PONR    = []
PONRVAR = []

#At every time point
for t in range(CURRENTGEN):
    myvec1 = 0
    myvec2 = []
    mylength = len(HAM_INF[t])
    #Get the avergae mu for UPNFL (i = 3; j = 1)
    for ind in range(mylength):
        myvec1 += (1- np.multiply(2,HAM_INF[t][ind][i][j])/DOMAINSIZE)
        myvec2.append((1- np.multiply(2,HAM_INF[t][ind][i][j])/DOMAINSIZE))
    PONRVAR.append(np.var(myvec2))
    PONR.append(myvec1/NUMIND)
    
    
#Effect of R on A  
i = 2-1
j = 1-1

RONA   = []
RONAVAR = []

#At every time point
for t in range(CURRENTGEN):
    myvec1 = 0
    myvec2 = []
    #Get the avergae mu for DOWNNFL (i = 3; j = 1)
    for ind in range(2000):
        myvec1 += (1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE)
        myvec2.append((1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE))
    RONAVAR.append(np.var(myvec2))
    RONA.append(myvec1/NUMIND)
    
    
    
    
#Effect of A on I  
i = 3-1
j = 2-1

AONI   = []
AONIVAR = []

#At every time point
for t in range(CURRENTGEN):
    myvec1 = 0
    myvec2 = []
    #Get the avergae mu for DOWNNFL (i = 3; j = 1)
    for ind in range(2000):
        myvec1 += (1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE)
        myvec2.append((1- np.multiply(2,HAM_TIME[t][ind][i][j])/DOMAINSIZE))
    AONIVAR.append(np.var(myvec2))
    AONI.append(myvec1/NUMIND)


import json
import random

# Generate a random number
random_number = random.randint(10000, 99999)



KS_VALUES = [array.tolist() for array in KS_VALUES]

with open(f'Ks_{random_number}.json', 'w') as f:
    json.dump(KS_VALUES, f)

with open(f'AVERAGEFH_{random_number}.json', 'w') as f:
    json.dump(AVERAGEFH, f)
with open(f'AVERAGEFP_{random_number}.json', 'w') as f:
    json.dump(AVERAGEFP, f)

with open(f'VARFITH_{random_number}.json', 'w') as f:
    json.dump(VARFITH, f)
with open(f'VARFITP_{random_number}.json', 'w') as f:
    json.dump(VARFITP, f)

with open(f'IONB_{random_number}.json', 'w') as f:
    json.dump(IONB, f)
with open(f'IONBVAR_{random_number}.json', 'w') as f:
    json.dump(IONBVAR, f)

with open(f'UPONRECEPT_{random_number}.json', 'w') as f:
    json.dump(UPONRECEPT, f)
with open(f'UPONRECEPTVAR_{random_number}.json', 'w') as f:
    json.dump(UPONRECEPTVAR, f)

with open(f'DOWNONRECEPT_{random_number}.json', 'w') as f:
    json.dump(DOWNONRECEPT, f)
with open(f'DOWNONRECEPTVAR_{random_number}.json', 'w') as f:
    json.dump(DOWNONRECEPTVAR, f)

with open(f'UPACT_{random_number}.json', 'w') as f:
    json.dump(UPACT, f)
with open(f'UPACTVAR_{random_number}.json', 'w') as f:
    json.dump(UPACTVAR, f)

with open(f'DOWNACT_{random_number}.json', 'w') as f:
    json.dump(DOWNACT, f)
with open(f'DOWNACTVAR_{random_number}.json', 'w') as f:
    json.dump(DOWNACTVAR, f)

RATEINPUT = [array.tolist() for array in RATEINPUT]
RATEOUPUT = [array.tolist() for array in RATEOUPUT]

with open(f'RATEINPUT_{random_number}.json', 'w') as f:
    json.dump(RATEINPUT, f)
with open(f'RATEOUPUT_{random_number}.json', 'w') as f:
    json.dump(RATEOUPUT, f)
    
with open(f'PONR_{random_number}.json', 'w') as f:
    json.dump(PONR, f)
with open(f'PONRVAR_{random_number}.json', 'w') as f:
    json.dump(PONRVAR, f)
    
with open(f'RONA_{random_number}.json', 'w') as f:
    json.dump(RONA, f)
with open(f'RONAVAR_{random_number}.json', 'w') as f:
    json.dump(RONAVAR, f)
    
with open(f'AONI_{random_number}.json', 'w') as f:
    json.dump(AONI, f)
with open(f'AONIVAR_{random_number}.json', 'w') as f:
    json.dump(AONIVAR, f)


