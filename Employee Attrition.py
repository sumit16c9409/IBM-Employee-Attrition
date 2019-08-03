
# coding: utf-8

# ## Importing required packages

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from bokeh.plotting import figure, show, output_file
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from math import pi
from bokeh.transform import cumsum
from bokeh.palettes import Category20c


# ## Setting the random seed so that the result for reproducibility

# In[2]:


np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


# ## Initializing Variables

# In[3]:


le = preprocessing.LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')


# ## Loading the Dataset

# In[4]:


dataset =pd.read_csv("Employee Attrition.csv")


# ## Encoding Categorical features

# In[5]:


def transform(feature):
    dataset[feature]=le.fit_transform(dataset[feature])


# ## Encoding all the columns

# In[6]:


columns =dataset.select_dtypes(include='object')
for col in columns.columns:
    transform(col)


# ## Feature Scaling

# In[7]:


scaler=StandardScaler()
dataset=dataset.astype(float)
X=scaler.fit_transform(dataset.drop(['Attrition','EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1))
Y=dataset['Attrition'].values


# ## One Hot Encoding the target Variable

# In[8]:


Y=to_categorical(Y)


# ## Dividing it into train and test set

# In[9]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# ## Building the Keras model

# In[10]:


model=Sequential()
model.add(Dense(input_dim=30,units=8,activation='relu'))
model.add(Dense(units=20,activation='relu'))
model.add(Dense(units=2,activation='sigmoid'))


# ## Defining loss and oprimizer 

# In[11]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# ## Training the Model fit

# In[12]:


history= model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,verbose=1)


# ## Evaluating the model

# In[13]:


model.evaluate(x_test,y_test)


# ## Train Loss vs Test Loss

# In[15]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# ## Train Accuracy vs Test Accuracy

# In[16]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# ## Reading the dataset again for Plots

# In[17]:


orignalDataset=pd.read_csv("Employee Attrition.csv")


# ## Defining colors for graphs

# In[18]:


def getColors(number):
    colors = ['#5E759F','#C58A6A','#6D9C71','#AA6263','#837CA8','#8D7867']
    colors = colors[0:number]
    return colors


# ## Function to plot Box Chart

# In[19]:


def boxChart(group,xrange,yrange):
    cats = group
    yy = xrange
    g = yrange
    for i, l in enumerate(cats):
        yy[g == l] += i // 2
    df = pd.DataFrame(dict(score=yy, group=g))

    # find the quartiles and IQR for each category
    groups = df.groupby('group')
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    # find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
    out = groups.apply(outliers).dropna()

    # prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = []
        outy = []
        for keys in out.index:
            outx.append(keys[0])
            outy.append(out.loc[keys[0]].loc[keys[1]])

    p = figure(tools="", background_fill_color="#efefef", x_range=cats, toolbar_location=None)

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.score = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'score']),upper.score)]
    lower.score = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'score']),lower.score)]

    # stems
    p.segment(cats, upper.score, cats, q3.score, line_color="black")
    p.segment(cats, lower.score, cats, q1.score, line_color="black")

    # boxes
    p.vbar(cats, 0.3, q2.score, q3.score, fill_color="#E08E79", line_color="black")
    p.vbar(cats, 0.3, q1.score, q2.score, fill_color="#3B8686", line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(cats, lower.score, 0.2, 0.01, line_color="black")
    p.rect(cats, upper.score, 0.2, 0.01, line_color="black")

    #outliers
    if not out.empty:
        p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.grid.grid_line_width = 2
    p.xaxis.major_label_text_font_size="12pt"
    output_file("age_attrition.html", title="Attrition based on Age")
    show(p)


# ## Function to plot Bar Chart

# In[20]:


def barChart(xrange,yrange,colors,title):
    output_file("total_attrition.html")
    attrition = xrange
    counts = yrange
    source = ColumnDataSource(data=dict(attrition=attrition, counts=counts, color=colors))
    p = figure(x_range=attrition, y_range=(0,max(counts)+int(0.25*max(counts))), plot_height=350, title=title,
               toolbar_location=None, tools="")
    p.vbar(x='attrition', top='counts', width=0.9, color='color', legend="attrition", source=source)
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_left"
    show(p)


# ## Function to plot Line Chart

# In[21]:


def lineChart(xrange,yrange):
    output_file("line.html")
    p = figure(plot_width=400, plot_height=400)
    # add a line renderer
    p.line(xrange, yrange, line_width=2)
    show(p)


# ## Function to plot Pie Chart¶

# In[22]:


def pieChart(x,title):
    output_file("pie.html")
    data = pd.Series(x).reset_index(name='value').rename(columns={'index':'item'})
    data['angle'] = data['value']/data['value'].sum() * 2*pi
    data['color'] = getColors(len(x))
    p = figure(plot_height=350, title=title, toolbar_location=None,
               tools="hover", tooltips="@item: @value", x_range=(-0.5, 1.0))
    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='item', source=data)
    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    show(p)


# ## Calculating different metrics for Graph

# In[23]:


attritionGroup = orignalDataset['Attrition'].unique().tolist()
attritionGroupReverse = attritionGroup[::-1]
ageList = orignalDataset['Age'].tolist()
attritionList = orignalDataset['Attrition'].tolist()

businessGroup = orignalDataset['BusinessTravel'].unique().tolist()
businessList = orignalDataset['BusinessTravel'].tolist()

attritionDeptList = orignalDataset[orignalDataset['Attrition'] == attritionGroup[0]]['Department'].tolist()
departmentGroup = orignalDataset['Department'].unique().tolist()

yesAttritionFrame = orignalDataset[orignalDataset['Attrition'] == attritionGroup[0]]
warnings.simplefilter("ignore")


# ## Charts analysed on entire organization

# In[24]:


#Bar Chart showing Total attrition
barChart(attritionGroup,[attritionList.count(attritionGroup[0]),attritionList.count(attritionGroup[1])],getColors(len(attritionGroup)),"Total Attrition")


# In[25]:


#Box Plot showing attrition based on Age
boxChart(attritionGroupReverse,ageList,attritionList)


# In[26]:


#Creating Gender List and group
genderList= orignalDataset[orignalDataset['Attrition']==attritionGroup[0]]['Gender'].tolist()
genderGroup=list(set(genderList))
genderGroupCount = [genderList.count(genderGroup[0]),genderList.count(genderGroup[1])]
#Bar Chart of Attrition based on Gender
barChart(genderGroup,genderGroupCount,getColors(len(genderGroup)),'Attrition based on Gender')


# In[27]:


#Pie Chart showing Attrition based on Department
x = {}
for i in range(len(departmentGroup)):
    x[departmentGroup[i]] = attritionDeptList.count(departmentGroup[i])
pieChart(x,"Attrition based on department")


# In[28]:


#Creating marital Status List and group
maritalStatusList= orignalDataset[orignalDataset['Attrition']==attritionGroup[0]]['MaritalStatus'].tolist()
maritalStatusGroup=list(set(maritalStatusList))
maritalStatusGroupCount = [maritalStatusList.count(maritalStatusGroup[0]),maritalStatusList.count(maritalStatusGroup[1]),maritalStatusList.count(maritalStatusGroup[2])]
#Bar Chart of Attrition based on Marital Status
barChart(maritalStatusGroup,maritalStatusGroupCount,getColors(len(maritalStatusGroup)),'Attrition based on Marital Status')


# ## Charts analysed on Reserach and Development Department

# In[29]:


#Attrition in Reserach and development Department
rdAttritionFrame = yesAttritionFrame[orignalDataset['Department'] == 'Research & Development']


# In[30]:


#Attrition in Research and Development based on amount of travel
travelList = rdAttritionFrame['BusinessTravel'].tolist()
travelGroup = orignalDataset['BusinessTravel'].unique().tolist()
barChart(travelGroup,[travelList.count(travelGroup[0]),travelList.count(travelGroup[1]),travelList.count(travelGroup[2])],getColors(len(travelGroup)),"Attrition based on Business Travel")


# In[31]:


#Creating OverTime List and group
overTimeList= rdAttritionFrame['OverTime'].tolist()
overTimeGroup=list(set(overTimeList))
overTimeGroupCount = [overTimeList.count(overTimeGroup[0]),overTimeList.count(overTimeGroup[1])]
#Bar Chart of Attrition based on Overtime
barChart(overTimeGroup,overTimeGroupCount,getColors(len(overTimeGroup)),"Attrition based on overtime")


# In[32]:


#Creating JobInvolvement List and Group
jobInvolvementList= rdAttritionFrame['JobInvolvement'].tolist()
jobInvolvementGroup=list(set(jobInvolvementList))
jobInvolvementGroupName = ['Low','Medium','High','Very High']
jobInvolvementGroupCount = [jobInvolvementList.count(jobInvolvementGroup[0]),jobInvolvementList.count(jobInvolvementGroup[1]),jobInvolvementList.count(jobInvolvementGroup[2]),jobInvolvementList.count(jobInvolvementGroup[3])]
#Pie Chart of Attrition based on JobInvolvement
x = {}
for i in range(len(jobInvolvementGroup)):
    x[jobInvolvementGroupName[i]] = jobInvolvementList.count(jobInvolvementGroup[i])
pieChart(x,"Attrition based on Job Involvement")


# In[33]:


#Grouping values based on PercentSalaryHike for Research and development
bins = pd.cut(rdAttritionFrame['PercentSalaryHike'], [10, 15, 20, 25])
percentHikeGroupList= rdAttritionFrame.groupby(bins)['PercentSalaryHike'].agg(['count'])['count'].tolist()
percentageGroup = ['10-15%','15-20%','20-25%']
#plotting pie chart based on PercentSalaryHike for Research and development
x = {}
for i in range(len(percentageGroup)):
    x[percentageGroup[i]] = percentHikeGroupList[i]
pieChart(x,"Attrition based on percentage Hike")


# In[34]:


#Grouping values based on Performance Rating
performanceRatingList= rdAttritionFrame['PerformanceRating'].tolist()
performanceRatingGroup=[1,2,3,4]
performanceRatingGroupName = ['Low','Good','Excellent','Outstanding']
performanceRatingGroupCount = [performanceRatingList.count(performanceRatingGroup[0]),performanceRatingList.count(performanceRatingGroup[1]),performanceRatingList.count(performanceRatingGroup[2]),performanceRatingList.count(performanceRatingGroup[3])]
#Bar Chart of Attrition based on Performance Rating
barChart(performanceRatingGroupName,performanceRatingGroupCount,getColors(len(performanceRatingGroup)),"Attrition based on Performance Rating")


# ## Charts analysed on Sales Department

# In[35]:


#Attrition in Sales Department
salesAttritionFrame = yesAttritionFrame[orignalDataset['Department'] == 'Sales']


# In[36]:


#Attrition in Sales based on amount of travel
travelList = salesAttritionFrame['BusinessTravel'].tolist()
travelGroup = orignalDataset['BusinessTravel'].unique().tolist()
barChart(travelGroup,[travelList.count(travelGroup[0]),travelList.count(travelGroup[1]),travelList.count(travelGroup[2])],getColors(len(travelGroup)),"Attrition based on Business Travel")


# In[37]:


#Creating OverTime List and group
overTimeList= salesAttritionFrame['OverTime'].tolist()
overTimeGroup=list(set(overTimeList))
overTimeGroupCount = [overTimeList.count(overTimeGroup[0]),overTimeList.count(overTimeGroup[1])]
#Bar Chart of Attrition based on Overtime
barChart(overTimeGroup,overTimeGroupCount,getColors(len(overTimeGroup)),"Attrition based on overtime")


# In[38]:


#Creating JobInvolvement List and Group
jobInvolvementList= salesAttritionFrame['JobInvolvement'].tolist()
jobInvolvementGroup=list(set(jobInvolvementList))
jobInvolvementGroupName = ['Low','Medium','High','Very High']
jobInvolvementGroupCount = [jobInvolvementList.count(jobInvolvementGroup[0]),jobInvolvementList.count(jobInvolvementGroup[1]),jobInvolvementList.count(jobInvolvementGroup[2]),jobInvolvementList.count(jobInvolvementGroup[3])]
#Pie Chart of Attrition based on JobInvolvement
x = {}
for i in range(len(jobInvolvementGroup)):
    x[jobInvolvementGroupName[i]] = jobInvolvementList.count(jobInvolvementGroup[i])
pieChart(x,"Attrition based on Job Involvement")


# In[39]:


#Grouping values based on PercentSalaryHike
bins = pd.cut(salesAttritionFrame['PercentSalaryHike'], [10, 15, 20, 25])
percentHikeGroupList= salesAttritionFrame.groupby(bins)['PercentSalaryHike'].agg(['count'])['count'].tolist()
percentageGroup = ['10-15%','15-20%','20-25%']
#plotting pie chart based on PercentSalaryHike for Research and development
x = {}
for i in range(len(percentageGroup)):
    x[percentageGroup[i]] = percentHikeGroupList[i]
pieChart(x,"Attrition based on percentage Hike")


# In[40]:


#Grouping values based on Performance Rating
performanceRatingList= salesAttritionFrame['PerformanceRating'].tolist()
performanceRatingGroup=[1,2,3,4]
performanceRatingGroupName = ['Low','Good','Excellent','Outstanding']
performanceRatingGroupCount = [performanceRatingList.count(performanceRatingGroup[0]),performanceRatingList.count(performanceRatingGroup[1]),performanceRatingList.count(performanceRatingGroup[2]),performanceRatingList.count(performanceRatingGroup[3])]
#Bar Chart of Attrition based on Performance Rating
barChart(performanceRatingGroupName,performanceRatingGroupCount,getColors(len(performanceRatingGroup)),"Attrition based on Performance Rating")


# ## Charts analysed on Human Resource Department

# In[41]:


#Attrition in Reserach and development Department
hrAttritionFrame = yesAttritionFrame[orignalDataset['Department'] == 'Human Resources']


# In[42]:


#Attrition based on amount of travel
travelList = hrAttritionFrame['BusinessTravel'].tolist()
travelGroup = orignalDataset['BusinessTravel'].unique().tolist()
barChart(travelGroup,[travelList.count(travelGroup[0]),travelList.count(travelGroup[1]),travelList.count(travelGroup[2])],getColors(len(travelGroup)),"Attrition based on Business Travel")


# In[43]:


#Creating OverTime List and group
overTimeList= hrAttritionFrame['OverTime'].tolist()
overTimeGroup=list(set(overTimeList))
overTimeGroupCount = [overTimeList.count(overTimeGroup[0]),overTimeList.count(overTimeGroup[1])]
#Bar Chart of Attrition based on Overtime
barChart(overTimeGroup,overTimeGroupCount,getColors(len(overTimeGroup)),"Attrition based on overtime")


# In[44]:


#Creating JobInvolvement List and Group
jobInvolvementList= hrAttritionFrame['JobInvolvement'].tolist()
jobInvolvementGroup=list(set(jobInvolvementList))
jobInvolvementGroupName = ['Low','Medium','High','Very High']
jobInvolvementGroupCount = [jobInvolvementList.count(jobInvolvementGroup[0]),jobInvolvementList.count(jobInvolvementGroup[1]),jobInvolvementList.count(jobInvolvementGroup[2]),jobInvolvementList.count(jobInvolvementGroup[3])]
#Pie Chart of Attrition based on JobInvolvement
x = {}
for i in range(len(jobInvolvementGroup)):
    x[jobInvolvementGroupName[i]] = jobInvolvementList.count(jobInvolvementGroup[i])
pieChart(x,"Attrition based on Job Involvement")


# In[45]:


#Grouping values based on PercentSalaryHike
bins = pd.cut(hrAttritionFrame['PercentSalaryHike'], [10, 15, 20, 25])
percentHikeGroupList= hrAttritionFrame.groupby(bins)['PercentSalaryHike'].agg(['count'])['count'].tolist()
percentageGroup = ['10-15%','15-20%','20-25%']
#plotting pie chart based on PercentSalaryHike for Research and development
x = {}
for i in range(len(percentageGroup)):
    x[percentageGroup[i]] = percentHikeGroupList[i]
pieChart(x,"Attrition based on percentage Hike")


# In[46]:


#Grouping values based on Performance Rating
performanceRatingList= hrAttritionFrame['PerformanceRating'].tolist()
performanceRatingGroup=[1,2,3,4]
performanceRatingGroupName = ['Low','Good','Excellent','Outstanding']
performanceRatingGroupCount = [performanceRatingList.count(performanceRatingGroup[0]),performanceRatingList.count(performanceRatingGroup[1]),performanceRatingList.count(performanceRatingGroup[2]),performanceRatingList.count(performanceRatingGroup[3])]
#Bar Chart of Attrition based on Performance Rating
barChart(performanceRatingGroupName,performanceRatingGroupCount,getColors(len(performanceRatingGroup)),"Attrition based on Performance Rating")


# ## Charts analysed on entire company

# In[47]:


#Grouping values based on YearsSinceLastPromotion
lastPromotionYearList = yesAttritionFrame['YearsSinceLastPromotion'].tolist()
lastPromotionYearGroup = list(set(lastPromotionYearList))
lastPromotionYearGroupCount = []
for i in range(len(lastPromotionYearGroup)):
    lastPromotionYearGroupCount.append(lastPromotionYearList.count(lastPromotionYearGroup[i]))
#lineChart showing attrition pattern with number of years since last promotion
lineChart(lastPromotionYearGroup,lastPromotionYearGroupCount)


# In[48]:


#Grouping values based on Distance
bins = pd.cut(yesAttritionFrame['DistanceFromHome'], [0, 10, 20, 30])
distanceGroupList= yesAttritionFrame.groupby(bins)['DistanceFromHome'].agg(['count'])['count'].tolist()
#plotting bar chart based on distance for Research and evelopment
barChart(['0-10','10-20','20-30'],distanceGroupList,getColors(len(distanceGroupList)),"Attrition based on Distance")


# In[49]:


#Creating Education Group
educationList = yesAttritionFrame['Education'].tolist()
educationGroup= list(set(educationList))
educationGroupName = ['Below College','College','Bachelor','Master','Doctor']
educationGroupCount = [educationList.count(educationGroup[0]),educationList.count(educationGroup[1]),educationList.count(educationGroup[2]),educationList.count(educationGroup[3]),educationList.count(educationGroup[4])]
#Pie chart showing Attrition based on Education for research and development
x = {}
for i in range(len(educationGroup)):
    x[educationGroupName[i]] = educationList.count(educationGroup[i])
pieChart(x,"Attrition based on Education")


# In[50]:


#Creating EnvironmentSatisfaction Group
envSatList=yesAttritionFrame['EnvironmentSatisfaction'].tolist()
envSatGroup=list(set(envSatList))
envSatGroupName = ['Low','Medium','High','Very High']
envSatGroupCount = [envSatList.count(envSatGroup[0]),envSatList.count(envSatGroup[1]),envSatList.count(envSatGroup[2]),envSatList.count(envSatGroup[3])]
#Bar Chart of Attrition based on EnvironmentSatisfaction
barChart(envSatGroupName,envSatGroupCount,getColors(len(envSatGroup)),"Attrition based on Environment Satisfaction")


# In[51]:


#Grouping values based on RelationshipSatisfaction
relationshipSatisfactionList= yesAttritionFrame['RelationshipSatisfaction'].tolist()
relationshipSatisfactionGroup= list(set(relationshipSatisfactionList))
relationshipSatisfactionGroupName = ['Low','Medium','High','Very High']
relationshipSatisfactionGroupCount = [relationshipSatisfactionList.count(relationshipSatisfactionGroup[0]),relationshipSatisfactionList.count(relationshipSatisfactionGroup[1]),relationshipSatisfactionList.count(relationshipSatisfactionGroup[2]),relationshipSatisfactionList.count(relationshipSatisfactionGroup[3])]
#Bar Chart of Attrition based on Performance Rating
barChart(relationshipSatisfactionGroupName,relationshipSatisfactionGroupCount,getColors(len(relationshipSatisfactionGroup)),"Attrition based on Relatinship Satisfaction")


# In[52]:


#Grouping values based on StockOptions
stockOptionList = yesAttritionFrame['StockOptionLevel'].tolist()
stockOptionGroup = list(set(stockOptionList))
stockOptionGroupName = ['Low','Medium','High','Very High']
stockOptionGroupCount = [stockOptionList.count(stockOptionGroup[0]),stockOptionList.count(stockOptionGroup[1]),stockOptionList.count(stockOptionGroup[2]),stockOptionList.count(stockOptionGroup[3])]
#Bar Chart of Attrition based on StockOptions
barChart(stockOptionGroupName,stockOptionGroupCount,getColors(len(stockOptionGroupName)),"Attrition based on Stock Option")


# In[53]:


#Grouping values based on WorkLifeBalance
workLifeBalanceList = yesAttritionFrame['WorkLifeBalance'].tolist()
workLifeBalanceGroup = list(set(workLifeBalanceList))
workLifeBalanceGroupName = ['Bad','Good','Better','Best']
workLifeBalanceGroupCount = [workLifeBalanceList.count(workLifeBalanceGroup[0]),workLifeBalanceList.count(workLifeBalanceGroup[1]),workLifeBalanceList.count(workLifeBalanceGroup[2]),workLifeBalanceList.count(workLifeBalanceGroup[3])]
#Bar Chart of Attrition based on StockOptions
barChart(workLifeBalanceGroupName,workLifeBalanceGroupCount,getColors(len(workLifeBalanceGroup)),"Attrition based on Work Life balance")


# In[54]:


#Grouping values based on YearsInCurrentRole
yearsInCurrentRoleList = yesAttritionFrame['YearsInCurrentRole'].tolist()
yearsInCurrentRoleGroup = list(set(yearsInCurrentRoleList))
yearsInCurrentRoleGroupCount = []
for i in range(len(yearsInCurrentRoleGroup)):
    yearsInCurrentRoleGroupCount.append(yearsInCurrentRoleList.count(yearsInCurrentRoleGroup[i]))
#lineChart showing attrition pattern with number of years
lineChart(yearsInCurrentRoleGroup,yearsInCurrentRoleGroupCount)


# In[55]:


#Grouping values based on YearsWithCurrManager
yearsWithCurrManagerList = yesAttritionFrame['YearsWithCurrManager'].tolist()
yearsWithCurrManagerGroup = list(set(yearsWithCurrManagerList))
yearsWithCurrManagerGroupCount = []
for i in range(len(yearsWithCurrManagerGroup)):
    yearsWithCurrManagerGroupCount.append(yearsWithCurrManagerList.count(yearsWithCurrManagerGroup[i]))
#lineChart showing attrition pattern with number of years
lineChart(yearsWithCurrManagerGroup,yearsWithCurrManagerGroupCount)


# In[56]:


#Percentage of females leaving the company
femalePercenatge = genderGroupCount[0]/int(orignalDataset[orignalDataset['Gender'] == 'Female']['Gender'].count())
print("Percentage of Female attrition =",int(femalePercenatge *100),"%")


# In[57]:


#Percentage of males leaving the company
malePercenatge = genderGroupCount[1]/int(orignalDataset[orignalDataset['Gender'] == 'Male']['Gender'].count())
print("Percentage of Male attrition =",int(malePercenatge *100),"%")

