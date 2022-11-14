import numpy as np
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

for i in range(1,6):
    f=open("exp"+str(i)+"_30_MATRIX1.txt","r")
    stringa=f.read()
    f.close()
    x=(stringa.replace("["," ")).replace("]","")
    f=open("exp"+str(i)+"_30_MATRIX1.txt","w")
    f.write(x)
    f.close()
for i in range(1,6):
    f=open("exp"+str(i)+"_30_MATRIX2.txt","r")
    stringa=f.read()
    f.close()
    x=(stringa.replace("["," ")).replace("]","")
    f=open("exp"+str(i)+"_30_MATRIX2.txt","w")
    f.write(x)
    f.close()
for i in range(1,6):
    f=open("exp"+str(i)+"_30_MSA.txt","r")
    stringa=f.read()
    f.close()
    x=(stringa.replace("["," ")).replace("]","")
    f=open("exp"+str(i)+"_30_MSA.txt","w")
    f.write(x)
    f.close()
for i in range(1,6):
    f=open("exp"+str(i)+"_30_MDS.txt","r")
    stringa=f.read()
    f.close()
    x=(stringa.replace("["," ")).replace("]","")
    f=open("exp"+str(i)+"_30_MDS.txt","w")
    f.write(x)
    f.close()

mca1=0
mca2=0

msa=0
mds=0

for i in range(1,6):
    f=open("SPEC_LEVEL/exp"+str(i)+"_30.txt","r")
    lines=f.readlines()
    mca1+=float(lines[17].split("  ")[1])
    mca2+=float(lines[35].split("  ")[1])
    print(mca2)

    #msa+=float(lines[16].split("  ")[1])
    #mds+=float(lines[18].split("  ")[1])
mca1=mca1/5
mca2=mca2/5

#msa=msa/5
#mds=mds/5
#print("MSA: ",msa)
print("MCA1: ",mca1)
print("MCA2: ",mca2)

#print("MDS: ",mds)
f.close()
dev_std_mca1=0
dev_std_mca2=0

#dev_std_msa=0
#dev_std_mds=0

for i in range(1,6):
    f=open("SPEC_LEVEL/exp"+str(i)+"_30.txt","r")
    lines=f.readlines()
    ca1=float(lines[17].split("  ")[1])
    ca2=float(lines[35].split("  ")[1])

    #sa=float(lines[16].split("  ")[1])
    #ds=float(lines[18].split("  ")[1])
    dev_std_mca1+=np.power(ca1-mca1,2)
    dev_std_mca2+=np.power(ca2-mca2,2)
    #dev_std_msa+=np.power(sa-msa,2)
    #dev_std_mds+=np.power(ds-mds,2)
dev_std_mca1=np.sqrt(dev_std_mca1/5).round(decimals=3)
dev_std_mca2=np.sqrt(dev_std_mca2/5).round(decimals=3)

#dev_std_msa=np.sqrt(dev_std_msa/5).round(decimals=3)
#dev_std_mds=np.sqrt(dev_std_mds/5).round(decimals=3)
f.close()


file=open("./MEAN_DEV_STD.txt","w")
mean_matrix=np.zeros([7,7])
for i in range (1,6):
    mat=np.loadtxt("exp"+str(i)+"_30_MATRIX1.txt")
    mean_matrix+=mat
mean_matrix=(mean_matrix/5).round(decimals=3)
print("MEAN MATRIX:\n",mean_matrix)
dev_std_matrix=np.zeros([7,7])

for i in range (1,6):
    mat=np.loadtxt("exp"+str(i)+"_30_MATRIX1.txt")
    dev_std_matrix+=np.power(mat-mean_matrix,2)
dev_std_matrix=np.sqrt(dev_std_matrix/5).round(decimals=3)
print("DEV STD MATRIX:\n",dev_std_matrix)

mean_matrix2=np.zeros([7,7])
for i in range (1,6):
    mat=np.loadtxt("exp"+str(i)+"_30_MATRIX2.txt")
    mean_matrix2+=mat
mean_matrix2=(mean_matrix2/5).round(decimals=3)
print("MEAN MATRIX2:\n",mean_matrix2)
dev_std_matrix2=np.zeros([7,7])

for i in range (1,6):
    mat=np.loadtxt("exp"+str(i)+"_30_MATRIX1.txt")
    dev_std_matrix2+=np.power(mat-mean_matrix2,2)
dev_std_matrix2=np.sqrt(dev_std_matrix2/5).round(decimals=3)
print("DEV STD MATRIX2:\n",dev_std_matrix2)

mean_seg_accuracies=np.zeros(7)
for i in range (1,6):
    accuracies=np.loadtxt("exp"+str(i)+"_30_MSA.txt")
    mean_seg_accuracies+=accuracies
mean_seg_accuracies=(mean_seg_accuracies/5).round(decimals=3)
print("MEAN SEG ACCURACIES:\n",mean_seg_accuracies)
msa=np.mean(mean_seg_accuracies)

dev_std_seg_accuracies=np.zeros(7)

for i in range (1,6):
    accuracies=np.loadtxt("exp"+str(i)+"_30_MSA.txt")
    dev_std_seg_accuracies+=np.power(accuracies-mean_seg_accuracies,2)
dev_std_seg_accuracies=np.sqrt(dev_std_seg_accuracies/5).round(decimals=3)
print("DEV STD SEG ACCURACIES:\n",dev_std_seg_accuracies)
dev_std_msa=np.mean(dev_std_seg_accuracies)


mean_dice_scores=np.zeros(7)
for i in range (1,6):
    dice_scores=np.loadtxt("exp"+str(i)+"_30_MDS.txt")
    mean_dice_scores+=dice_scores
mean_dice_scores=(mean_dice_scores/5).round(decimals=3)
print("MEAN DICE SCORES:\n",mean_dice_scores)
mds=np.mean(mean_dice_scores)
dev_std_dice_scores=np.zeros(7)

for i in range (1,6):
    dice_scores=np.loadtxt("exp"+str(i)+"_30_MDS.txt")
    dev_std_dice_scores+=np.power(dice_scores-mean_dice_scores,2)
dev_std_dice_scores=np.sqrt(dev_std_dice_scores/5).round(decimals=3)
print("DEV STD DICE SCORES:\n",dev_std_dice_scores)
dev_std_mds=np.mean(dev_std_dice_scores)

mean_matrix_total=np.matrix([mean_matrix.diagonal(), mean_seg_accuracies,mean_dice_scores])
dev_std_matrix_total=np.matrix([dev_std_matrix.diagonal(), dev_std_seg_accuracies,dev_std_dice_scores])



#####PLOTTING
class_dict = {0: 'homo',
              1: 'speck',
              2: 'nucleo',
              3: 'centro',
              4: 'golgi',
              5: 'numem',
              6: 'mitsp'}
fig, ax = plot_confusion_matrix(conf_mat=mean_matrix,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,class_names=class_dict.values())
plt.title("MEAN PATTERN CLASSIFICATION MATRIX")
#plt.show()
plt.savefig("RESULTS/MEAN_PATTERN_CLASSIFICATION_MATRIX2.png")

#############
labels = ['homo','speck','nucleo','centro','golgi','numem','mitsp']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


#############

fig, ax = plt.subplots()
im = ax.imshow(mean_matrix)

# Show all ticks and label them with the respective list entries

ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        if i==j:
            text = ax.text(j, i, str(mean_matrix[i, j].round(decimals=3)) +"\n+/-\n"+str(dev_std_matrix[i,j].round(decimals=2)),
                        ha="center", va="center", color="w", size=11, weight="bold")
        else:
            text = ax.text(j, i, str(mean_matrix[i, j].round(decimals=3)) +"\n+/-\n"+str(dev_std_matrix[i,j].round(decimals=2)),
                        ha="center", va="center", color="w", size=11)


ax.set_title("MEAN +/- DEV STD PATTERN CLASSIFICATION MATRIX")

fig.tight_layout()
#plt.show()
plt.savefig("RESULTS/MEAN+DEV_STD_PATTERN_CLASSIFICATION_MATRIX.png")

############

############# MEAN MATRIX 

fig, ax = plt.subplots()
im = ax.imshow(mean_matrix)
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        if i==j:
            text = ax.text(j, i, mean_matrix[i, j],
                        ha="center", va="center", color="w", size=11, weight="bold")
        else:
            text = ax.text(j, i, mean_matrix[i, j],
                        ha="center", va="center", color="w", size=11)


ax.set_title("MEAN PATTERN CLASSIFICATION MATRIX")

fig.tight_layout()
#plt.show()
plt.savefig("RESULTS/MEAN_PATTERN_CLASSIFICATION_MATRIX.png")

############

############ DEV STD MATRIX
fig, ax = plt.subplots()
im = ax.imshow(dev_std_matrix)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        if i==j:
            text = ax.text(j, i, dev_std_matrix[i, j],
                        ha="center", va="center", color="w", size=11, weight="bold")
        else:
            text = ax.text(j, i, dev_std_matrix[i, j],
                        ha="center", va="center", color="w", size=11)

ax.set_title("DEV STD PATTERN CLASSIFICATION MATRIX")

fig.tight_layout()
#plt.show()
plt.savefig("RESULTS/DEV_STD_PATTERN_CLASSIFICATION_MATRIX.png")
#######


############
fig, ax = plt.subplots()
im = ax.imshow(mean_matrix_total)
metrics=["MCA","MSA","MDS"]
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metrics)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(metrics)):
    for j in range(len(labels)):
        text = ax.text(j, i, mean_matrix_total[i, j],
                       ha="center", va="center", color="w",weight="bold")

ax.set_title("MEAN METRICS PER CLASS")

fig.tight_layout()
#plt.show()
plt.savefig("RESULTS/MEAN_METRICS_PER_CLASS.png")
#######

############
fig, ax = plt.subplots()
im = ax.imshow(dev_std_matrix_total)
metrics=["MCA","MSA","MDS"]
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metrics)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(metrics)):
    for j in range(len(labels)):
        text = ax.text(j, i, dev_std_matrix_total[i, j],
                       ha="center", va="center", color="w",weight="bold")

ax.set_title("DEV STD METRICS PER CLASS")

fig.tight_layout()
#plt.show()
plt.savefig("RESULTS/DEV_STD_METRICS_PER_CLASS.png")
#######

#############

fig, ax = plt.subplots()
im = ax.imshow(mean_matrix_total)

# Show all ticks and label them with the respective list entries

ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metrics)


# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(metrics)):
    for j in range(len(labels)):
            text = ax.text(j, i, str(mean_matrix_total[i, j].round(decimals=3)) +"\n+/-\n"+str(dev_std_matrix_total[i,j].round(decimals=2)),
                        ha="center", va="center", color="w", size=11, weight="bold")


ax.set_title("MEAN +/- DEV STD METRICS PER CLASS")

fig.tight_layout()
#plt.show()
plt.savefig("RESULTS/MEAN+DEV_STD_METRICS_PER_CLASS.png")

############


#############
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_matrix.diagonal(), width, label='Mean class accuracy')
rects2 = ax.bar(x + width/2, dev_std_matrix.diagonal(), width, label='Dev std class accuracy')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Class accuracies')
ax.set_title('Class accuracies')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

#plt.show()
plt.savefig("RESULTS/CLASS_ACCURACIES.png")
############


#############
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_seg_accuracies, width, label='Mean seg accuracy')
rects2 = ax.bar(x + width/2, dev_std_seg_accuracies, width, label='Dev std seg accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Segmentation accuracies')
ax.set_title('Segmentation accuracies by group ')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend()

fig.tight_layout()

#plt.show()
plt.savefig("RESULTS/SEGMENTATION_ACCURACIES.png")

############



############
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_dice_scores, width, label='Mean seg accuracy')
rects2 = ax.bar(x + width/2, dev_std_dice_scores, width, label='Dev std seg accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Dice scores per class')
ax.set_title('Dice scores by class')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend()

fig.tight_layout()

#plt.show()
plt.savefig("RESULTS/DICE_SCORES.png")

############
file.write("MEAN PATTERN CLASSIFICATION MATRIX:\n")
file.write(np.array2string(mean_matrix))
file.write("\n")
file.write("DEV STD PATTERN CLASSIFICATION MATRIX:\n")
file.write(np.array2string(dev_std_matrix))
file.write("\n")
file.write("MEAN PATTERN CLASSIFICATION MATRIX2:\n")
file.write(np.array2string(mean_matrix2))
file.write("\n")
file.write("DEV STD PATTERN CLASSIFICATION MATRIX2:\n")
file.write(np.array2string(dev_std_matrix2))
file.write("\n")
file.write("MEAN CLASS SEGMENTATION ACCURACIES:\n")
file.write(np.array2string(mean_seg_accuracies))
file.write("\n")
file.write("DEV STD CLASS SEGMENTATION ACCURACIES:\n")
file.write(np.array2string(dev_std_seg_accuracies))
file.write("\n")
file.write("MEAN CLASS DICE SCORES:\n")
file.write(np.array2string(mean_dice_scores))
file.write("\n")
file.write("DEV STD CLASS DICE SCORES:\n")
file.write(np.array2string(dev_std_dice_scores))
file.write("\nMSA: "+str(msa))
file.write("\nMCA1: "+str(mca1))
file.write("\nMCA2: "+str(mca2))

file.write("\nMDS: "+str(mds))
file.write("\nDEV STD MSA: "+str(dev_std_msa))
file.write("\nDEV STD MCA1: "+str(dev_std_mca1))
file.write("\nDEV STD MCA2: "+str(dev_std_mca2))
file.write("\nDEV STD MDS: "+str(dev_std_mds))
file.close()





