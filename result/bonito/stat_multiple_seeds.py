# %%
import pandas as pd
from scipy.stats import ranksums

result = pd.read_csv("bonito_multiple_seeds.tsv", sep='\t')
mean = result.mean()
std = result.std()
print(result)
print(mean)
print(std)
read_pvalue = ranksums(result["read_conv"],result["read_markonv"])[1]
print(read_pvalue)
assembly_pvalue = ranksums(result["assembly_conv"],result["assembly_markonv"])[1]
print(assembly_pvalue)
# %%
