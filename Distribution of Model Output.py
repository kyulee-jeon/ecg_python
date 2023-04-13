# necessary: y_prob, y_true 
##############################################################################################
import matplotlib.patches as mpatches

# stemi, control
df = pd.DataFrame({'Probs': y_prob[:,1], 'Label': y_true[:,1]})
df.loc[df['Label']==0, 'Label'] = 'Control'
df.loc[df['Label']==1, 'Label'] = 'STEMI'

stemi = np.array(df[df['Label']=='STEMI']['Probs'])
control = np.array(df[df['Label']=='Control']['Probs'])


# create color patches for the labels
stemi_patch = mpatches.Patch(color='red', label='STEMI')
control_patch = mpatches.Patch(color='black', label='Control')

sns.histplot(data = stemi, bins = 5, binwidth = 0.05, color = "Red", legend="STEMI", stat = "probability", alpha=0.5, multiple='dodge')
sns.histplot(data = control, bins = 5, binwidth = 0.05,  color = "Black", legend="Control", stat = "probability", alpha=0.5, multiple='dodge')      
# add the legend with the color patches
plt.legend(handles=[stemi_patch, control_patch])

# create and add the labels with the color patches
#plt.text(0.05, -0.2, 'Control:', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
#plt.text(0.15, -0.2, 'STEMI:', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.gca().add_patch(mpatches.Rectangle((0.11, -0.25), 0.05, 0.05, alpha=0.5, facecolor='red'))
plt.gca().add_patch(mpatches.Rectangle((0.01, -0.25), 0.05, 0.05, alpha=0.5, facecolor='black'))

# set the x and y limits and labels
plt.xlim([-0.05,1.05])  
plt.ylim([0.0, 1.]) 
plt.xlabel("Model Output (Probability of STEMI)", size=10)    
plt.ylabel("Proportion of Sample Per Group", size=10) 

# show the plot
plt.show()
