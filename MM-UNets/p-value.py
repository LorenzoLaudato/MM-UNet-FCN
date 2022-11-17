import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
import statistics
from statsmodels.stats.multitest import multipletests



csv_mca = pd.read_csv('mca.csv')
print(csv_mca)

E1 = csv_mca["E1"]
E1 = E1.to_numpy() 
E2 = csv_mca["E2"]
E2 = E2.to_numpy() 

E3 = csv_mca["E3"]
E3 = E3.to_numpy()

E4 = csv_mca["E4"]
E4 = E4.to_numpy() 

E5 = csv_mca["E5"]
E5 = E5.to_numpy() 

meanE1 = statistics.mean(E1)
devE1 = statistics.pstdev(E1)
print(meanE1)
print(devE1)

meanE2 = statistics.mean(E2)
devE2 = statistics.pstdev(E2)
print(meanE2)
print(devE2)
meanE3=statistics.mean(E3)
devE3 = statistics.pstdev(E3)
print(meanE3)
print(devE3)


meanE4=statistics.mean(E4)
devE4 = statistics.pstdev(E4)
print(meanE4)
print(devE4)

meanE5=statistics.mean(E5)
devE5 = statistics.pstdev(E5)
print(meanE5)
print(devE5)

##BETTER  XX
res = wilcoxon(E2,E1)
p21 = res.pvalue

res = wilcoxon(E2,E3)
p23 = res.pvalue

res = wilcoxon(E2,E4)
p24 = res.pvalue

res = wilcoxon(E2,E5)
p25 = res.pvalue

pvalues=[p21,p23,p24,p25]

print(pvalues)

pvalues_final = multipletests(pvalues, method='bonferroni')[1]
print(pvalues_final)



