# Employee Turnover Prediction

![Employee Turnover](/images/turnover.png)

## Dataset Overview

**About the Data**
The dataset comprises various features, both categorical and numerical, aiming to predict whether employees will leave the company in the future. These attributes provide valuable insights for HR decision-making and employee retention strategies.

**Dataset Details:**
- **Education:** Education level of the employee.
- **JoiningYear:** Year in which the employee joined the company.
- **City:** City where the employee's office is located.
- **PaymentTier:** Payment tier categorization (1: Highest, 2: Mid Level, 3: Lowest).
- **Age:** Current age of the employee.
- **Gender:** Gender of the employee.
- **EverBenched:** Whether the employee has ever been kept out of projects for 1 month or more.
- **ExperienceInCurrentDomain:** The number of years of experience employees have in their current field.
- **Label:**
  - **LeaveOrNot:** Binary indicator of whether the employee is predicted to leave the company in the future (1: Leaves, 0: Stays).

**Dataset Source:**
The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset/data).

## Project Structure

- **1. Problem Description:**
    - The primary objective is to develop a predictive model that accurately forecasts employee turnover in the future. The model's insights will empower HR teams to proactively address retention challenges and implement strategic initiatives to enhance employee satisfaction, ultimately reducing attrition.

## 2. Exploratory Data Analysis (EDA)

- **Analysis of Categorical Variables:**

    - **Education Distribution:**
        - Visual representation of the distribution of education levels.
        - ![Education Distribution](/images/education.png)

    - **City Distribution:**
        - Exploration of the distribution of the cities where employees' offices are located.
        - ![City Distribution](/images/city.png)

    - **Gender Distribution:**
        - Examination of the distribution of genders among employees.
        - ![Gender Distribution](/images/gender.png)

    - **Ever Benched Distribution:**
        - Analysis of the distribution of employees who have been benched.
        - ![Ever Benched Distribution](/images/everbenched.png)

    - **Joining Year Distribution:**
        - Visualizing the distribution of the years when employees joined the company.
        - ![Joining Year Distribution](/images/year.png)

    - **PaymentTier Distribution:**
        - Distribution of payment tier categorization.
        - ![PaymentTier Distribution](/images/paymenttier.png)
              
- **Analysis of Numerical Variables:**

    - **Age Distribution:**
        - Investigating the distribution of ages among employees.
        - ![Age Distribution](/images/age.png)

    - **Experience in Current Domain Distribution:**
        - Examining the distribution of years of experience in the current domain.
        - ![Experience Distribution](/images/experience.png)

### Target Variable Analysis:

#### Target Summary with Numeric:

- **Age:**
    - Leave: 29.052500
    - Stay: 29.571896

- **Experience in Current Domain:**
    - Leave: 2.840000
    - Stay: 2.940059

*Note: Numeric variables' analysis involved examining the mean values of the target variable.*

#### Target Summary with Categorical:

- **Education:**
    - Bachelors: 3601
    - Masters: 873
    - PHD: 179

- **City:**
    - Bangalore: 2228
    - New Delhi: 1157
    - Pune: 1268

- **Gender:**
    - Female: 1875
    - Male: 2778

- **Ever Benched:**
    - No: 4175
    - Yes: 478

- **LeaveOrNot:**
    - Leave: 1600
    - Stay: 3053

- **Joining Year:**
    - 2012: 504
    - 2013: 669
    - 2014: 699
    - 2015: 781
    - 2016: 525
    - 2017: 1108
    - 2018: 367

- **PaymentTier:**
    - 1: 243
    - 2: 918
    - 3: 3492

*Note: Categorical variables' analysis involved examining the count values of the target variable.*
