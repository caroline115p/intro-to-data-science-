#!/usr/bin/env python
# coding: utf-8

# # Assignment 3: Confidence Intervals & The Bootstrap
# 
# ### Instructions
# 
# This assignment is much like the last.
# 
# * This assignment includes some tests to help you make sure that your implementation is correct.  When you see a cell with `assert` commands, these are tests.
# 
# * Once you have completed the assignment, delete the cells which have these `assert` commands.  You will not need them.
# 
# * When you are done and have answered all the questions, convert this notebook to a .py file using `File > Download as > Python (.py)`.  Name your submission `assignment3.py` and submit it to OWL.
# 
# Failure to comply may resilt in you not earning full marks for your assignment.  We want you to earn full marks!  Please follow these instructions carefully.

# # Question 1
# 
# ### Part A
# 
# Recall from theclture that a $199(1-\alpha)\%$ confidence interval for the mean is 
# 
# $$ \bar{x} \pm  t_{1-\alpha/2, n-1} \dfrac{\hat{\sigma}}{\sqrt{n}} $$
# 
# Where $ t_{1-\alpha/2, n-1}$ is the appropiorate quantile of a Student's t distribution with $n-1$ degrees of freedom.  When $\alpha = 0.05$ and when $n$ is big enough, $ t_{1-\alpha/2, n-1} \approx 1.96$.  
# 
# Write a function called `confidence_interval` which takes as it's argument an array of data called `data` and returns two things:
# 
# * An estimated mean of `data`, and 
# 
# * The lower and upper bounds of the 95% confidence interval for the mean of `data`.  Ensure these are returned in a numpy array of shape (2,)

# In[2]:


#It's dangerous to go alone.  Take these
import numpy as np
import scipy.stats as t
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import pandas as pd


def confidence_interval(data):
    '''
    Function to compute confidence interval (ci).
    
    Inputs:
    data - ndarray.  Data to be used to compute the interval
    
    Outputs:
    estimated_mean - float.  The mean of the data
    bounds - array. An array of length 2 where bounds[0] is the lower bound and bounds[1] is the upper bound for the ci.
    '''
    ### BEGIN SOLUTION
    a = np.array(data)
    my_df = len(data) - 1 
    critval = t.t.ppf(0.975, df = my_df)
    stderr = np.std(a)/np.sqrt(len(data))
    bounds = [np.mean(a) - critval*stderr, np.mean(a) + critval*stderr]
    estimated_mean = np.mean(a)
    

    ### END SOLUTION
    
    return estimated_mean, bounds
confidence_interval([-1,0,1])[0]


# ### Tests
# 
# Tests are to make sure you've implemented the solution correctly.  If these tests pass without any `AsserstionError`'s, then you can be confident that you've implemented the solution as expected.
# 
# Once you're happy with your implementation, delete the cell below.

# In[ ]:





# ### Part B
# 
# The "95% confidence interval" is named so because the long term relative frequency of these estimtors containing the true estimand is 95%.  That is to say **if I construct 95% confidence intervals for the sample mean again and again from the same data generating mechanism, 95% of the intervals I construct will contain the population mean**.
# 
# Write a function called `ci_simulation` that runs some simulations to show this is the case.  From a standard normal distirbution, sample 25 observations and construct a confidence interval.  Do this 20 times and plot the intervals using `matplotlib.pyplot.errorbar`. Save your plot under the name `ci_simulation.png`.  Color the bar red if the confidence interval does not caputre the true mean and blue if it does.  If you are unfamilliar with `matplotlib.pyplot.errorbar`, I highly suggest reading Matplotlib's excellent documentation.

# In[16]:


import numpy as np


def ci_simulation():
    # Reporducibility.  Do not change this!
    np.random.seed(3)
    
    # Create the figure.
    fig, ax = plt.subplots(dpi = 120) 

    # If the interval crosses this line, it should be blue, else red.
    ax.axhline(0, color = 'k')
    
    ### BEGIN SOLUTION
    for i in range (20): #20 times using for loop 
        mean,(confidence_interval_low,confidence_interval_up) = confidence_interval(np.random.normal(0,1,25))
        if confidence_interval_low > 0  :
            plt.errorbar (i, mean, confidence_interval_up - mean, fmt ='o',color = "red")
        elif confidence_interval_up < 0 : 
            plt.errorbar (i, mean, confidence_interval_up - mean, fmt ='o',color = "red")
        else:
            plt.errorbar (i, mean, confidence_interval_up - mean, fmt ='o',color = "blue")
    ### END SOLUTION
    # This function does not have to return anything
    return None
ci_simulation()


# ### Part C
# 
# If you haven't changed the random seed from 3 and if you implemented the solution correctly, you should two red intervals.
# 
# Answer the following below in 1-2 sentences:
# 
# 1) How many red intervals did we expect to see?  What is your justifiation for this?
# 
# 2) If there is a discrepency between the number of observed and expected red intervals, what explains this difference?
# 
# 

# In[4]:


#1) We expected to see 1 because the 5% of 20 is 1 so we will expect to see the 1 intervals.
#2) That is because of the random sampling. We used normal random sampling method in the error bar so the number we observed and expected have different red intervals. 


# ### Part D
# 
# How many samples would we need in order to ensure that our constructed confidence interval is approximately 0.1 units long? 
# 
# Write a function called `num_propper_length` which takes as its only argument an integer `n`.  `num_propper_length` should simulate 1000 datasets of length `n`, compute confidence intervals for those datasets, compute the lengths of those intervals, and then returns the number of intervals which are no longer than 0.1 units.
# 
# Determine how many samples you need (that is, compute `n`).  Set this as your default argument for `n` in `num_propper_length`.

# In[15]:


def num_propper_length(n=1600):
    '''
    Function to simulate how many confidence intervals would be longer than 0.1 units long if
    we sampled n observatiosn from a standard normal.
    
    Inputs:
        n - int.  Number of draws to make from the standard normal
        
    Outputs:
        prop_long_enough - float (b   etween 0 and 1).  Proportion of consutrcuted intervals which are no longer than 0.1.
    '''
    # For reproducibility.  Don't change this!
    np.random.seed(0)
    
    ### BEGIN SOLUTION
    for i in range (1000):
        mean = confidence_interval(np.random.normal(0,1,n))[0]
        confidence_interval_low = confidence_interval(np.random.normal(0,1,n))[1][0]
        confidence_interval_up = confidence_interval(np.random.normal(0,1,n))[1][1]
      # have to make sure if is greater than 0.1 or not 
    ### END SOLUTION
    return prop_long_enough

num_propper_length(1600)


# ### Tests
# 
# Tests are to make sure you've implemented the solution correctly.  If these tests pass without any `AsserstionError`'s, then you can be confident that you've implemented the solution as expected.
# 
# Once you're happy with your implementation, delete the cell below.

# In[ ]:





# In[ ]:





# ### Part E
# If you chose the right `n`, you should find that 891 (or approximately 89%) of your intervals are longer than 0.1.  
# 
# Why is this?  Answer below in 1-2 sentences.
# 
# ### Answer:
# 
# Random sampling.

# In[ ]:


#If I chose the right n then we will get 891 because if this is asymmetrical confidence interval then we are dealing with a Normal distribution


# ---

# ## Question 2
# 
# ### Part A
# The dataset `hockey_stats.csv` contains information about information about hockey draftees.  We'll use this data to investigate the relationship between height and age on weight.  Load it into python using pandas.
# 
# Load in the `hockey_draftees_train.csv` data into pandas.  Fit a linear model of weight (`wt`) explained by height (`ht`) and age(`age`).  Call your fitted model `model`.

# In[27]:


### BEGIN SOLUTION
import statsmodels.formula.api as sm

import pandas as pd
data = pd.read_csv("hockey_draftees_train.csv")
display(data)


model = sm.ols('wt ~ ht + age', data = data).fit()


model.summary()

### END SOLUTION


# ### Part B
# 
# Print out the R-squared for this model
# 

# In[28]:




print('Training rsquared is ',model.rsquared)


# ### Part C
# 
# Now, let's see how well our model performs out of sample.  Load in the `hockey_draftees_test.csv` file into a dataframe.  Use your `model` to make predictions, and print the r-squared on the out of sample (oos) data.

# In[42]:


### BEGIN SOLUTION
import statsmodels.formula.api as sm

hockey_data = pd.read_csv("hockey_draftees_test.csv")
display(hockey_data)

hockey_model = sm.ols('wt ~ ht + age ', data = hockey_data).fit()
rsquared_oos = hockey_model.rsquared 

### END SOLUTION

print('Out of sample rsquared is ', rsquared_oos)


# ### Part D
# 
# A point estimate of the rsquared is nice, but what we really want is uncertainty estimates.  For that, we need a confidence interval.  To estimate how uncertain we are in our out of sample r-squared, let's use the bootstrap.
# 
# Write a function called `bootstrap` which takes three arguments:
# 
# * `data`, which is a dataframe, and 
# * `model` which is an statsmodel ols model. `data` should look the the data `model` was trained on so that we can use `model` to make predictions on `data`.
# * `numboot` which is an integer denoting how many bootstrap replications to perform.
# 
# Write `bootstrap` to perform bootstrap resampling for the out of sample r-squared.  You can use `pd.DataFrame.sample` with `replace = True` to perform the resampling.
# 
# `bootstrap` should return a numpy array of bootstraped rsquared values.
# 
# 

# In[94]:


def bootstrap(data, model, numboot):
    '''
    Function to bootstrap the r-squared for a linear model
    
    Inputs:
        data - dataframe.  Data on which model can predict.
        model - statsmodel ols.  Linear model of weight explained by height and age.  Can predict on data.
        numboot - int.  Number of bootstrap replications.
    
    Outputs:
        bootstrapped_rsquared - array.  An array of size (numboot, ) which contains oos bootstrapped rsquared values.
    
    '''
    ### BEGIN SOLUTION
    bootstrapped_rsquared = []
    n = len(data)
    np.random.seed(0) #each of the bootstrap will generate the new set of number 
    for i in range(numboot):
        d = data.sample(n, replace = True)
        model =  sm.ols('wt ~ ht + age',data).fit()
        bootstrapped_rsquared.append(model.rsquared)
    ### END SOLUTION

    return bootstrapped_rsquared


# In[ ]:





# ### Part E
# 
# Use your `bootstrap` function to plot 10,000 bootstrap replicates as a histogram.

# In[97]:


import seaborn as sns
### BEGIN SOLUTION
data = pd.read_csv("hockey_draftees_test.csv")

bm = bootstrap(data,model, 10000)
sns.distplot(bm)

### END SOLUTION


# ### Part F
# 
# Use your bootstrap replicates to estimates to obtain a bootstrapped 95% confidence interval.  Call the upper confidence bound `ci_upper` and the lower confidence bound `ci_lower`.

# In[152]:


### BEGIN SOLUTIOn
lbm = bootstrap(data,model,10000)
loss_boot_ci = np.quantile(lbm, [0.025,0.975])
ci_lower = loss_boot_ci[0]
ci_upper = loss_boot_ci[1]


### END SOLUTION

print('My confidence interval is between', ci_lower, ' and ', ci_upper)


# In[ ]:




