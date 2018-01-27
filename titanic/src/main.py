"""
This script analysis data from  survivers of the Titanic.
The goal is to make a predictive model that would allow us to
know if a given passanger has survived or not
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Importing data
train=pd.read_csv('../data/train.csv')
test=pd.read_csv('../data/test.csv')

# First look into our training data
print(train.info())
print(train.head())

"""
Data looks good at this point. There are some missing values
but nothing that would compromise our analysis
At this point there are some questions that we would like to 
answer. Such as:
    1- Which type of passangers survived the most those of 1st class, 2nd class or 3rd class?
    2- Which type of person was most likely to survive: men, women or children?
    3- In general, people who bought a more expensive ticket had more chances to survive or not?
    4- Did it matter where people embarked the Titanic in terms of survival?

Once the abovementioned are answered we will have a more clear idea of the profile of passanger that was more 
likely to survive. To predict if a passanger has survived or not a logistic regression algorithm will be implemented
"""

# 1- Which type of passangers survived the most those of 1st class or those of 2nd class?

# first we will extract all those passangers from first and third class
first_class_passangers=train[train["Pclass"]==1]
second_class_passangers=train[train["Pclass"]==2]
third_class_passangers=train[train["Pclass"]==3]


handle=plt.hist([first_class_passangers["Survived"], second_class_passangers['Survived'], third_class_passangers['Survived']], bins=3)
plt.xlabel("Status")
plt.xticks([0.15,0.85],['Not survived','Survived'])
plt.ylabel("Number of passangers")
plt.legend(['First class','Second class', 'Third class'])
plt.title("Class vs Survival")
plt.show()

"""
In the last plot we can confirm some facts that were expected. The number of passangers of first class that survived is greater than those of the second class; however that the number
of passanger surviving in the 3rd class is greater than those of the 2nd class is unexpected. Nevertheless the magnitude of survivals for the three classes is comparable. On the contrary, the number of passangers who died in the third class is
of greater magnitude than those of second and first class, being the first class the one that had less passangers dead. 
"""
# Let's see which range of age for both classes survived the most. We will fill those entries without age with the median age
# Now we can retrieve for each of them the survivers and deceased
survivals_1_c=first_class_passangers[first_class_passangers["Survived"]==1]
survivals_2_c=second_class_passangers[second_class_passangers["Survived"]==1]
survivals_3_c=third_class_passangers[third_class_passangers["Survived"]==1]


age_surv1=survivals_1_c['Age'].fillna(survivals_1_c['Age'].median())
age_surv2=survivals_2_c['Age'].fillna(survivals_2_c['Age'].median())
age_surv3=survivals_3_c['Age'].fillna(survivals_3_c['Age'].median())

figure, axarr =plt.subplots(nrows=3, ncols=1)

axarr[0].hist(age_surv1,bins=20)
axarr[0].set_title('Survivers first class = %s'%age_surv1.size)
axarr[0].set_ylabel('Number of survivers')

axarr[1].hist(age_surv2,bins=20)
axarr[1].set_title('Survivers second class = %s'%age_surv2.size)
axarr[1].set_ylabel('Number of survivers')

axarr[2].hist(age_surv3,bins=20)
axarr[2].set_title('Survivers third class = %s'%age_surv3.size)
axarr[2].set_ylabel("Number of survivers")
axarr[2].set_xlabel('Age')
plt.show()


"""
In the last plot we can see different things. first, that the total number of survivers in first class is greater than in third class (as expected). Then, the age range of people 
in first class seems to be equally distributed between 18 to 55 years of age, whereas in third class almost no children between 7 and 12 survived. However there is a peak in the
number of survivals of children from 0 to 7 years and for those between 15 and 30.
From the 3rd class histogram it seems obvious that small children were favoured onto the saving boats in regard twith pre-adolescents and teenagers (also expected). Nevertheless this same
result can't be concluded from the 1st class histogram. Is it because children were not favoured (not likely) or because there were almost no children onboard on the 1st class (most likely)
Let's find out!
"""

# 2- Which type of person was most likely to survive: men, women or children?
# In this case we will consider children as any-gender person whose age is lower than 12 years of age and we will compare it w.r.t. the total number of passangers of that sex
survivals_1_c=survivals_1_c.dropna(axis=0,subset=['Age'])
first_class_passangers=first_class_passangers.dropna(axis=0, subset=['Age'])

survivals_2_c=survivals_2_c.dropna(axis=0,subset=['Age'])
second_class_passangers=second_class_passangers.dropna(axis=0, subset=['Age'])

survivals_3_c=survivals_3_c.dropna(axis=0,subset=['Age'])
third_class_passangers=third_class_passangers.dropna(axis=0, subset=['Age'])

gender_surv_1c=survivals_1_c['Sex']
gender_surv_1c[survivals_1_c['Age']<12]='children'

gender_1c=pd.DataFrame(first_class_passangers['Sex'], columns=['Sex'])
gender_1c[first_class_passangers['Age']<12]='children'
gender_1c["Sex_surv"]=gender_surv_1c


gender_surv_2c=survivals_2_c['Sex'].copy()
gender_surv_2c[survivals_2_c['Age']<12]='children'

gender_2c=pd.DataFrame(second_class_passangers['Sex'],columns=['Sex'])
gender_2c[second_class_passangers['Age']<12]='children'
gender_2c["Sex_surv"]=gender_surv_2c

gender_surv_3c=survivals_3_c['Sex'].copy()
gender_surv_3c[survivals_3_c['Age']<12]='children'

gender_3c=pd.DataFrame(third_class_passangers['Sex'], columns=['Sex'])
gender_3c[third_class_passangers['Age']<12]='children'
gender_3c['Sex_surv']=gender_surv_3c


figure, axarr=plt.subplots(nrows=3,ncols=1)

axarr[0].hist(gender_1c)
axarr[0].set_title('First class')
axarr[0].set_ylabel('Number of passangers')
axarr[0].legend()
axarr[1].hist(gender_2c)
axarr[1].set_title('Second class')
axarr[1].set_ylabel('Number of passangers')
axarr[2].hist(gender_3c)
axarr[2].set_title('Third class')
axarr[2].set_ylabel('Number of passangers')
axarr[2].set_xlabel('Gender')
plt.show()



















# 3- In general, people who bought a more expensive ticket had more chances to survive or not?


# 4- Did it matter where people embarked the Titanic in terms of survival?

