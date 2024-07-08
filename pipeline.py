

@transform_pandas(
    Output(rid="ri.vector.main.execute.52fa4c8f-b186-4f7b-b84f-fc86f07e91f1"),
    death_survivor_function_main=Input(rid="ri.vector.main.execute.d44a60f2-d36c-44f8-8a6a-7320e78c0395")
)
def death_curve_main( death_survivor_function_main):
    BOOTSTRAP_SURVIVAL_CURVES_FULL_DEATH_copied = death_survivor_function_main

    main_df = BOOTSTRAP_SURVIVAL_CURVES_FULL_DEATH_copied
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    set_output_image_type('svg')
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'blue', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'orange', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    ax.set_title('Mortality', fontsize=11)
    ax.set_ylabel('Cumulative Incidence (%)', fontsize=10)
    ax.set_xlabel('Day', fontsize=10)
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.7689b597-d68d-4adf-9adb-c1160626141e"),
    death_survivorfunction_trial1=Input(rid="ri.vector.main.execute.0958beed-149b-4f2e-8c78-cd63b2fe42c9")
)
def death_curve_trial1( death_survivorfunction_trial1):

    main_df = death_survivorfunction_trial1
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.6d84c49b-90bb-44b4-9878-e9a289ed8262"),
    death_survivorfunction_trial3=Input(rid="ri.vector.main.execute.a03e001d-20c9-4053-8e00-b6609496abc0")
)
def death_curve_trial3( death_survivorfunction_trial3):
    death_survivorfunction_trial1 = death_survivorfunction_trial3

    main_df = death_survivorfunction_trial1
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.4d8d5e79-cb99-48d4-a893-33e36c5622e4"),
    death_survivorfunction_trial2=Input(rid="ri.vector.main.execute.7cd03c4d-a7c6-4e03-91bb-0fffb2323160")
)
def death_curve_trial_2( death_survivorfunction_trial2):
    death_survivorfunction_trial1 = death_survivorfunction_trial2

    main_df = death_survivorfunction_trial1
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.38074363-e8a6-4343-8293-2d4c5830dd8e"),
    death_surv_function_unadjusted=Input(rid="ri.vector.main.execute.0e5761a6-c4db-4ce8-adc6-c0f3f05dee51")
)
def death_curve_unadjusted( death_surv_function_unadjusted):
    main_df = death_surv_function_unadjusted
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.e351da6f-8926-4cf0-8973-dd2e81ab0009"),
    death_surv_function_unadjusted_copied=Input(rid="ri.vector.main.execute.422db330-47da-4f83-bbc7-93b7c076516d")
)
def death_curve_unadjusted_t1( death_surv_function_unadjusted_copied):
    main_df = death_surv_function_unadjusted_copied
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.625e20e3-fd52-4098-910a-8bdf7b613410"),
    death_surv_function_unadjusted_copied_1=Input(rid="ri.vector.main.execute.6dee6b21-bcb5-4f1c-ade3-524a0964158c")
)
def death_curve_unadjusted_t2( death_surv_function_unadjusted_copied_1):
    main_df = death_surv_function_unadjusted_copied_1
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.67d410bc-185f-4fed-ae63-f813dd668836"),
    death_surv_function_unadjusted_copied_2=Input(rid="ri.vector.main.execute.b1d2dc99-ecc8-4ca0-8afd-0907ead088d2")
)
def death_curve_unadjusted_t3( death_surv_function_unadjusted_copied_2):
    main_df = death_surv_function_unadjusted_copied_2
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.0e5761a6-c4db-4ce8-adc6-c0f3f05dee51"),
    hosp_curve_unadjusted=Input(rid="ri.vector.main.execute.1afad117-be80-4b6c-8390-2dbbb29bd61b")
)
def death_surv_function_unadjusted( Analysis_dataset_combined_ate, hosp_curve_unadjusted):
    MAIN_HOSP_copied = hosp_curve_unadjusted
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_death_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'death90'], 
        # weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_death_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'death90'], 
            # weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.422db330-47da-4f83-bbc7-93b7c076516d"),
    hosp_curve_unadjusted_t1=Input(rid="ri.vector.main.execute.c666173b-1b6b-4ac7-892c-200353be7c1e")
)
def death_surv_function_unadjusted_copied( Analysis_dataset_combined_ate, hosp_curve_unadjusted_t1):
    MAIN_HOSP_copied = hosp_curve_unadjusted_t1
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(col('trial') == 1)

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_death_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'death90'], 
        # weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_death_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'death90'], 
            # weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.6dee6b21-bcb5-4f1c-ade3-524a0964158c"),
    hosp_curve_unadjusted_t2=Input(rid="ri.vector.main.execute.9f81885c-deea-4e96-9fda-6c32d2a4ac42")
)
def death_surv_function_unadjusted_copied_1( Analysis_dataset_combined_ate, hosp_curve_unadjusted_t2):
    MAIN_HOSP_copied = hosp_curve_unadjusted_t2
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(col('trial') == 2)

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_death_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'death90'], 
        # weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_death_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'death90'], 
            # weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.b1d2dc99-ecc8-4ca0-8afd-0907ead088d2"),
    hosp_curve_unadjusted_t3=Input(rid="ri.vector.main.execute.2d0f6de9-f95a-4f84-aade-ba6d58e3684a")
)
def death_surv_function_unadjusted_copied_2( Analysis_dataset_combined_ate, hosp_curve_unadjusted_t3):
    MAIN_HOSP_copied = hosp_curve_unadjusted_t3
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(col('trial') == 3)

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_death_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'death90'], 
        # weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_death_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'death90'], 
            # weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.d44a60f2-d36c-44f8-8a6a-7320e78c0395"),
    hosp_curve_main=Input(rid="ri.vector.main.execute.7e48237a-0972-460e-a967-432f3ed71e58")
)
def death_survivor_function_main( Analysis_dataset_combined_ate, hosp_curve_main):
    MAIN_HOSP_copied = hosp_curve_main
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_death_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'death90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_death_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'death90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.0958beed-149b-4f2e-8c78-cd63b2fe42c9"),
    hospital_curve_trial1=Input(rid="ri.vector.main.execute.5e3e1457-1259-4e7d-b86f-d162e00e4d8a")
)
def death_survivorfunction_trial1( Analysis_dataset_combined_ate, hospital_curve_trial1):
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(expr('trial == 1'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_death_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'death90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_death_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'death90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.7cd03c4d-a7c6-4e03-91bb-0fffb2323160"),
    hospital_curve_trial2=Input(rid="ri.vector.main.execute.1895745d-02a1-463c-ac38-5b8ae239ba02")
)
def death_survivorfunction_trial2( Analysis_dataset_combined_ate, hospital_curve_trial2):
    hospital_curve_trial1 = hospital_curve_trial2
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(expr('trial == 2'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_death_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'death90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_death_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'death90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.a03e001d-20c9-4053-8e00-b6609496abc0"),
    hospital_curve_trial3=Input(rid="ri.vector.main.execute.e70652d9-d16f-4dbf-a317-5920ada20455")
)
def death_survivorfunction_trial3( Analysis_dataset_combined_ate, hospital_curve_trial3):
    hospital_curve_trial1 = hospital_curve_trial3
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(expr('trial == 3'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_death_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'death90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_death_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'death90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.ce4a6615-0022-45f6-81e8-40a14c25a929"),
    hospital_surv_function_composite=Input(rid="ri.vector.main.execute.1f712ff7-5da5-4451-bdd8-bf60b2ac95a1")
)
def hosp_curve_composite( hospital_surv_function_composite):
    main_df = hospital_surv_function_composite
    
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.a72cb796-4c94-4d11-ac99-4c5c13409cfa"),
    hospital_surv_function_composite_copied_11=Input(rid="ri.vector.main.execute.95a6f69f-0efb-4bf0-ab48-4bdae64b57a2")
)
def hosp_curve_composite_copied_trial1( hospital_surv_function_composite_copied_11):
    main_df = hospital_surv_function_composite_copied_11
    
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    # set_output_image_type('svg')
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Initiator', 'Noninitiator'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    ax.set_title('Composite Outcome')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([1] + np.arange(7, 35, 7).tolist())
    ax.set_ylabel('Cumulative Incidence (%)')
    ax.set_xlabel('')
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.d5732145-3c84-4cd9-acb3-a18125e0e58f"),
    hospital_surv_function_composite_copied_12=Input(rid="ri.vector.main.execute.abc5a4d3-9b4d-4656-b2af-dfa1a17f9396")
)
def hosp_curve_composite_copied_trial2( hospital_surv_function_composite_copied_12):
    main_df = hospital_surv_function_composite_copied_12
    
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.3fcc2487-ffd6-4e59-957a-1f46f6619f89"),
    hospital_surv_function_composite_copied_13=Input(rid="ri.vector.main.execute.cc16d744-0adb-42a0-911f-50d6ed237b62")
)
def hosp_curve_composite_copied_trial3( hospital_surv_function_composite_copied_13):
    main_df = hospital_surv_function_composite_copied_13
    
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.27f1f617-776d-4a61-88bd-cdcf2a107b56"),
    hospital_surv_function_composite_unadjusted=Input(rid="ri.vector.main.execute.a29727f3-a3e1-4ee7-8fa1-61b764b40f80")
)
def hosp_curve_composite_unadjusted( hospital_surv_function_composite_unadjusted):
    main_df = hospital_surv_function_composite_unadjusted
    
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.7e48237a-0972-460e-a967-432f3ed71e58"),
    hospital_survivor_function_main=Input(rid="ri.vector.main.execute.75083aa5-93e5-4a08-acc9-6438886c4809")
)
def hosp_curve_main( hospital_survivor_function_main):
    BOOTSTRAP_SURVIVAL_CURVES_FULL_HOSP_UNADJUSTED_copied = hospital_survivor_function_main
    
    main_df = BOOTSTRAP_SURVIVAL_CURVES_FULL_HOSP_UNADJUSTED_copied
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    set_output_image_type('svg')
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'blue', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'orange', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    ax.set_title('Hospitalization', fontsize=11)
    ax.set_ylabel('Cumulative Incidence (%)', fontsize=10)
    ax.set_xlabel('', fontsize=10)
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()

    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.1afad117-be80-4b6c-8390-2dbbb29bd61b"),
    hospital_surv_function_unadjusted=Input(rid="ri.vector.main.execute.7dcce44c-583c-4149-b1f9-c93c61496521")
)
def hosp_curve_unadjusted( hospital_surv_function_unadjusted):
    main_df = hospital_surv_function_unadjusted
    
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.c666173b-1b6b-4ac7-892c-200353be7c1e"),
    hospital_surv_function_unadjusted_copied=Input(rid="ri.vector.main.execute.ddf2ac9f-be29-4b1e-b1cb-cc79c1422fef")
)
def hosp_curve_unadjusted_t1( hospital_surv_function_unadjusted_copied):
    main_df = hospital_surv_function_unadjusted_copied
    
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.9f81885c-deea-4e96-9fda-6c32d2a4ac42"),
    hospital_surv_function_unadjusted_copied_1=Input(rid="ri.vector.main.execute.403ba0ac-49bd-421e-bfba-ca2dd85beb46")
)
def hosp_curve_unadjusted_t2( hospital_surv_function_unadjusted_copied_1):
    main_df = hospital_surv_function_unadjusted_copied_1
    
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.2d0f6de9-f95a-4f84-aade-ba6d58e3684a"),
    hospital_surv_function_unadjusted_copied_2=Input(rid="ri.vector.main.execute.98bed0c6-e961-4e63-a360-701a15589aa2")
)
def hosp_curve_unadjusted_t3( hospital_surv_function_unadjusted_copied_2):
    main_df = hospital_surv_function_unadjusted_copied_2
    
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.5e3e1457-1259-4e7d-b86f-d162e00e4d8a"),
    hospital_survivorfunction_trial1=Input(rid="ri.vector.main.execute.91778d73-f8ce-4d1e-9032-2e84b4a701e5")
)
def hospital_curve_trial1( hospital_survivorfunction_trial1):

    main_df = hospital_survivorfunction_trial1
     
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.1895745d-02a1-463c-ac38-5b8ae239ba02"),
    hospital_survivorfunction_trial2=Input(rid="ri.vector.main.execute.d30861bd-04d1-4457-a63d-dd2aa74a1253")
)
def hospital_curve_trial2( hospital_survivorfunction_trial2):
    hospital_survivorfunction_trial1 = hospital_survivorfunction_trial2

    main_df = hospital_survivorfunction_trial1
     
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.e70652d9-d16f-4dbf-a317-5920ada20455"),
    hospital_survivorfunction_trial3=Input(rid="ri.vector.main.execute.ca0fbd07-15f5-4deb-ba9d-2c47b867dbeb")
)
def hospital_curve_trial3( hospital_survivorfunction_trial3):
    hospital_survivorfunction_trial1 = hospital_survivorfunction_trial3

    main_df = hospital_survivorfunction_trial1
     
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != -998).toPandas()
    df = df.set_index(['timeline','bootstrap'])
    df = df.rename_axis('treatment', axis=1)
    df = df.stack()
    df = pd.DataFrame(df)
    df.columns = ['surv']
    # df['cum_inc'] = 1 - df['surv']
    df['cum_inc'] = df['surv']
    df = df.reset_index(drop = False)
    print(df.head())

    ######## NEW CODE - REPEAT FOR THE OVERALL CURVE ##############
    df_overall = main_df.where(col('bootstrap') == -998).toPandas()
    df_overall = df_overall.set_index(['timeline','bootstrap'])
    df_overall = df_overall.rename_axis('treatment', axis=1)
    df_overall = df_overall.stack()
    df_overall = pd.DataFrame(df_overall)
    df_overall.columns = ['mean_surv']
    # df['cum_inc'] = 1 - df['surv']
    df_overall['cum_inc'] = df_overall['mean_surv'] # The function is already a cumulative incidence
    df_overall = df_overall.reset_index(drop = False)
    df_overall = df_overall.sort_values(by = ['treatment','timeline'])
    # print(df_overall)
    ###############################################################

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.groupby(['treatment', 'timeline']).agg(mean_surv = ('cum_inc', np.mean),
    ll = ('cum_inc', lower_quantile),
    ul = ('cum_inc', upper_quantile)
    )
    
    df = df.reset_index()

    ### NOW WE CAN PLOT
    fig, ax = plt.subplots(1,1, figsize = (11, 6))

    # Plot the curves for each group
    df_overall.query('treatment == "treatment"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'blue', drawstyle="steps-post") # Plot marginal survival curve (averaged) for treated group
    df_overall.query('treatment == "control"').plot(x = 'timeline', y = 'mean_surv', ax = ax, color = 'orange', drawstyle="steps-post") # Plot the averaged marginal survival curve for the control group
    ax.legend(['Treated', 'Untreated'])

    # Plot the CI - first for the treated group (using fill_between)
    ax.fill_between(x = df.loc[df['treatment'] == "treatment", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "treatment", 'll'], 
                    y2 = df.loc[df['treatment'] == "treatment", 'ul'], 
                    color = 'purple', alpha = 0.2, step = 'post')

    # PLot the CI for the control group
    ax.fill_between(x = df.loc[df['treatment'] == "control", 'timeline'], 
                    y1 = df.loc[df['treatment'] == "control", 'll'], 
                    y2 = df.loc[df['treatment'] == "control", 'ul'], 
                    color = 'pink', alpha = 0.2, step = 'post')

    ax.set_ylim([0.0, df['mean_surv'].max() + 0.05 * df['mean_surv'].max()])
    plt.show()

    ##### NEXT WE WANT TO CALCULATE THE PROBABILITY DIFFERENCE ON DAY 28 AND THE RISK RATIO ON DAY 28
    df = main_df.toPandas()
    df = df.set_index(['timeline','bootstrap'])
    # # Now calculate the probability difference
    # df['treatment'] = 1 - df['treatment']
    # df['control'] = 1 - df['control']
    df['treatment'] = df['treatment']
    df['control'] = df['control']
    df['risk_reduction'] = df['control'] - df['treatment']
    # Now take the risk ratio
    df['risk_ratio'] = df['treatment']/df['control']
    print(df.head())

    # Aggregate the curves by treatment and day; get the mean survival and the lower and upper limits
    df = df.query('bootstrap != -998') #### WE NEED TO ADD THIS
    df_statistics = df.groupby(['timeline']).agg(risk_reduction = ('risk_reduction', np.mean),
    risk_reduction_se = ('risk_reduction', np.std),
    risk_reduction_ll = ('risk_reduction', lower_quantile),
    risk_reduction_ul = ('risk_reduction', upper_quantile),
    # Get statistics for the risk ratio
    risk_ratio = ('risk_ratio', np.mean),
    risk_ratio_se = ('risk_ratio', np.std),
    risk_ratio_ll = ('risk_ratio', lower_quantile),
    risk_ratio_ul = ('risk_ratio', upper_quantile)
    )

    # Calculate the CI using the SE
    df_statistics['risk_ratio_lower95'] = df_statistics['risk_ratio'] - 1.96*df_statistics['risk_ratio_se']
    df_statistics['risk_ratio_upper95'] = df_statistics['risk_ratio'] + 1.96*df_statistics['risk_ratio_se']
    df_statistics = df_statistics.reset_index()
    ############################
    #### We need to swap the risk reduction and the risk ratio with the point estimate from the full sample
    control_cuminc = df_overall.loc[(df_overall['treatment'] == 'control') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    treatment_cuminc = df_overall.loc[(df_overall['treatment'] == 'treatment') & (df_overall['timeline'] == 28), 'cum_inc'].values[0]
    risk_reduction = control_cuminc - treatment_cuminc
    risk_ratio = treatment_cuminc/control_cuminc

    # substitute those values into the table
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_ratio'] = risk_ratio
    df_statistics.loc[(df_statistics['timeline'] == 28), 'risk_reduction'] = risk_reduction
    print(df_statistics.loc[df_statistics['timeline'] == 28, ['risk_ratio','risk_ratio_ll','risk_ratio_ul']])
    ############################

    return df_statistics

@transform_pandas(
    Output(rid="ri.vector.main.execute.1f712ff7-5da5-4451-bdd8-bf60b2ac95a1"),
    death_curve_trial3=Input(rid="ri.vector.main.execute.6d84c49b-90bb-44b4-9878-e9a289ed8262")
)
def hospital_surv_function_composite( Analysis_dataset_combined_ate, death_curve_trial3):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.withColumn('event90', expr('CASE WHEN event90 >= 1 THEN 1 ELSE event90 END'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'event90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'event90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.95a6f69f-0efb-4bf0-ab48-4bdae64b57a2"),
    hosp_curve_composite=Input(rid="ri.vector.main.execute.ce4a6615-0022-45f6-81e8-40a14c25a929")
)
def hospital_surv_function_composite_copied_11( Analysis_dataset_combined_ate, hosp_curve_composite):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.withColumn('event90', expr('CASE WHEN event90 >= 1 THEN 1 ELSE event90 END')).where(expr('trial == 1'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'event90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'event90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.abc5a4d3-9b4d-4656-b2af-dfa1a17f9396"),
    hosp_curve_composite_copied_trial1=Input(rid="ri.vector.main.execute.a72cb796-4c94-4d11-ac99-4c5c13409cfa")
)
def hospital_surv_function_composite_copied_12( Analysis_dataset_combined_ate, hosp_curve_composite_copied_trial1):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.withColumn('event90', expr('CASE WHEN event90 >= 1 THEN 1 ELSE event90 END')).where(expr('trial == 2'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'event90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'event90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.cc16d744-0adb-42a0-911f-50d6ed237b62"),
    hosp_curve_composite_copied_trial2=Input(rid="ri.vector.main.execute.d5732145-3c84-4cd9-acb3-a18125e0e58f")
)
def hospital_surv_function_composite_copied_13( Analysis_dataset_combined_ate, hosp_curve_composite_copied_trial2):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.withColumn('event90', expr('CASE WHEN event90 >= 1 THEN 1 ELSE event90 END')).where(expr('trial == 3'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'event90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'event90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.a29727f3-a3e1-4ee7-8fa1-61b764b40f80"),
    death_curve_trial3=Input(rid="ri.vector.main.execute.6d84c49b-90bb-44b4-9878-e9a289ed8262")
)
def hospital_surv_function_composite_unadjusted( Analysis_dataset_combined_ate, death_curve_trial3):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.withColumn('event90', expr('CASE WHEN event90 >= 1 THEN 1 ELSE event90 END'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'event90'], 
        # weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'event90'], 
            # weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.7dcce44c-583c-4149-b1f9-c93c61496521")
)
def hospital_surv_function_unadjusted( Analysis_dataset_combined_ate, hosp_curve_composite_copied_trial3):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_hospitalized_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'hospitalized90'], 
        # weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_hospitalized_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'hospitalized90'], 
            # weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.ddf2ac9f-be29-4b1e-b1cb-cc79c1422fef"),
    death_curve_unadjusted=Input(rid="ri.vector.main.execute.38074363-e8a6-4343-8293-2d4c5830dd8e")
)
def hospital_surv_function_unadjusted_copied( Analysis_dataset_combined_ate, death_curve_unadjusted):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(col('trial') == 1)

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_hospitalized_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'hospitalized90'], 
        # weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_hospitalized_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'hospitalized90'], 
            # weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.403ba0ac-49bd-421e-bfba-ca2dd85beb46"),
    death_curve_unadjusted_t1=Input(rid="ri.vector.main.execute.e351da6f-8926-4cf0-8973-dd2e81ab0009")
)
def hospital_surv_function_unadjusted_copied_1( Analysis_dataset_combined_ate, death_curve_unadjusted_t1):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(col('trial') == 2)

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_hospitalized_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'hospitalized90'], 
        # weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_hospitalized_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'hospitalized90'], 
            # weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.98bed0c6-e961-4e63-a360-701a15589aa2"),
    death_curve_unadjusted_t2=Input(rid="ri.vector.main.execute.625e20e3-fd52-4098-910a-8bdf7b613410")
)
def hospital_surv_function_unadjusted_copied_2( Analysis_dataset_combined_ate, death_curve_unadjusted_t2):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(col('trial') == 3)

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_hospitalized_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'hospitalized90'], 
        # weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_hospitalized_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'hospitalized90'], 
            # weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.75083aa5-93e5-4a08-acc9-6438886c4809")
)
def hospital_survivor_function_main( Analysis_dataset_combined_ate):
    Analysis_dataset_combined_1 = Analysis_dataset_combined_ate

    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_hospitalized_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'hospitalized90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_hospitalized_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'hospitalized90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.91778d73-f8ce-4d1e-9032-2e84b4a701e5"),
    death_curve_main=Input(rid="ri.vector.main.execute.52fa4c8f-b186-4f7b-b84f-fc86f07e91f1")
)
def hospital_survivorfunction_trial1( Analysis_dataset_combined_ate, death_curve_main):
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(expr('trial == 1'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_hospitalized_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'hospitalized90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_hospitalized_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'hospitalized90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.d30861bd-04d1-4457-a63d-dd2aa74a1253"),
    death_curve_trial1=Input(rid="ri.vector.main.execute.7689b597-d68d-4adf-9adb-c1160626141e")
)
def hospital_survivorfunction_trial2( Analysis_dataset_combined_ate, death_curve_trial1):
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(expr('trial == 2'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_hospitalized_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'hospitalized90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_hospitalized_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'hospitalized90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.ca0fbd07-15f5-4deb-ba9d-2c47b867dbeb")
)
def hospital_survivorfunction_trial3( Analysis_dataset_combined_ate, death_curve_trial_2):
    death_curve_trial2 = death_curve_trial_2
    
    weight_type = 'MMWS'
    bootstraps = 500

    import datetime
    ct = datetime.datetime.now()
    print('START TIME', ct)

    cr = Analysis_dataset_combined_ate.where(expr('trial == 3'))

    # How many unique person_ids are there? 
    unique_persons = cr.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    import random
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Create an empty list to store the survival curve data frames
    CIF_DF_LIST = []
    

    ####### FIT THE KM CURVE FOR THE FULL SAMPLE TO GET THE POINT ESTIMATE #######
    cumulative_incidence_functions_full = []
    km_full = KaplanMeierFitter()

    # Convert DF to Pandas
    cr_full = cr.toPandas()

    for group, group_label in zip([0, 1],['control','treatment']):

        # Fit the model to the full sample
        km_full.fit(cr_full.loc[cr_full['treatment'] == group, 'time_to_hospitalized_trunc90'], 
        event_observed = cr_full.loc[cr_full['treatment'] == group, 'hospitalized90'], 
        weights = cr_full.loc[cr_full['treatment'] == group, weight_type],
        label = group_label)

        # Save the CIF for the group
        CIF_DF_FULL = km_full.survival_function_

        # Append to the list of CIFs for this treatment group
        cumulative_incidence_functions_full.append(CIF_DF_FULL)

    # Append to the list
    CIF_DF_FULL = pd.concat(cumulative_incidence_functions_full, axis=1)
    CIF_DF_FULL['bootstrap'] = 999
    CIF_DF_LIST.append(CIF_DF_FULL)

    ##############################################################################

    for i in np.arange(0, bootstraps):
        
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Join to the main data frame; convert to Pandas
        cr_sample = sample_ids_df.join(cr, on = 'person_id', how = 'inner')
        cr_sample = cr_sample.toPandas()
        
        # Fit the KM curve
        cumulative_incidence_functions = []

        km = KaplanMeierFitter()

        for group, group_label in zip([0, 1],['control','treatment']):

            km.fit(cr_sample.loc[cr_sample['treatment'] == group, 'time_to_hospitalized_trunc90'], 
            event_observed = cr_sample.loc[cr_sample['treatment'] == group, 'hospitalized90'], 
            weights = cr_sample.loc[cr_sample['treatment'] == group, weight_type],
            label = group_label)

            CIF = km.survival_function_
            cumulative_incidence_functions.append(CIF)

        # Join the cumulative incidences of the groups together (axis=1)
        CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
        CIF_DF['bootstrap'] = i
        CIF_DF_LIST.append(CIF_DF)

    final = pd.concat(CIF_DF_LIST)
    final = 1 - final # Convert to cumulative incidence

    ct = datetime.datetime.now()
    print('END TIME', ct)

    return final.reset_index()

    

