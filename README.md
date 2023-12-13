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

#### Target Summary with Categorical:

- **Education:**
    - Bachelors: 0.313524
    - Masters: 0.487973
    - PHD: 0.251397

- **City:**
    - Bangalore: 0.267056
    - New Delhi: 0.316335
    - Pune: 0.503943

- **Gender:**
    - Female: 0.471467
    - Male: 0.257739

- **Ever Benched:**
    - No: 0.331257
    - Yes: 0.453975

- **LeaveOrNot:**
    - 0: 0.0
    - 1: 1.0

- **Joining Year:**
    - 2012: 0.216270
    - 2013: 0.334828
    - 2014: 0.247496
    - 2015: 0.407170
    - 2016: 0.222857
    - 2017: 0.268051
    - 2018: 0.986376

- **PaymentTier:**
    - 1: 0.366255
    - 2: 0.599129
    - 3: 0.275200

- **3. Prepare Data for Model Training:**
    - Extract only those rows in the column leaveornot who are either Stay or Leave as value.
    - Split the data in a two-step process which finally leads to the distribution of 60% train, 20% validation, and 20% test sets with random seed to `11`.
    - Prepare target variable `leaveornot` by converting it from categorical to binary, where 0 represents `Stay` and 1 represents `Leave`.
    - Finally delete the target variable from the train/val/test dataframe.
