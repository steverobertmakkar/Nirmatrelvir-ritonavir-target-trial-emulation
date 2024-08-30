

@transform_pandas(
    Output(rid="ri.vector.main.execute.35f765bd-f032-4ade-a996-b2acfd2d86b4"),
    composite_KMcurve_main=Input(rid="ri.vector.main.execute.f26742b9-4902-4790-8303-fc2f5f9cdbda"),
    death_KMcurve_main=Input(rid="ri.vector.main.execute.a0e75935-7689-4229-9050-2286d8523139"),
    hosp_KMcurve_main=Input(rid="ri.vector.main.execute.374213bc-f702-4f4b-ac08-51cd73e74ff9")
)
def Revision_Main(hosp_KMcurve_main, death_KMcurve_main, composite_KMcurve_main):

    # Main result:
    # Hospitalization pooled
    # Trials 1-3
    # Death pooled
    # Trials 1-3

    ### Unadjusted
    # Hospitalization pooled
    # Trials 1-3
    # Death pooled
    # Trials 1-3

    # Composite
    # Pooled
    # Trials 1-3

    df1 = hosp_KMcurve_main.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Hospitalization')).withColumn('category', lit('Pooled')).withColumn('rank', lit(1))
    df2 = death_KMcurve_main.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Mortality')).withColumn('category', lit('Pooled')).withColumn('rank', lit(5))
    df3 = composite_KMcurve_main.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Composite')).withColumn('category', lit('Pooled')).withColumn('rank', lit(9))
    

    final = df1.union(df2).union(df3)

    return final
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.354b3cf7-b898-42be-9098-596b1e51210e"),
    hosp_KMcurve_t1=Input(rid="ri.vector.main.execute.0bda773b-7b18-4b66-931a-e31de4e0bed7")
)
def Revision_Trial1(hosp_KMcurve_t1, death_KMcurve_t1, composite_KMcurve_t1):

    # Main result:
    # Hospitalization pooled
    # Trials 1-3
    # Death pooled
    # Trials 1-3

    ### Unadjusted
    # Hospitalization pooled
    # Trials 1-3
    # Death pooled
    # Trials 1-3

    # Composite
    # Pooled
    # Trials 1-3

    df1 = hosp_KMcurve_t1.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Hospitalization')).withColumn('category', lit('Trial 1')).withColumn('rank', lit(2))
    df2 = death_KMcurve_t1.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Mortality')).withColumn('category', lit('Trial 1')).withColumn('rank', lit(6))
    df3 = composite_KMcurve_t1.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Composite')).withColumn('category', lit('Trial 1')).withColumn('rank', lit(10))
    

    final = df1.union(df2).union(df3)

    return final
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.4c613d50-3d88-4d5b-abd1-0eb4006d02e4"),
    composite_KMcurve_t2=Input(rid="ri.vector.main.execute.72f99aa7-04c3-4736-8523-addb99dfee0a"),
    death_KMcurve_t2=Input(rid="ri.vector.main.execute.1e9edb45-8d80-4b0b-84b1-679c491536c2")
)
def Revision_Trial2(hosp_KMcurve_t2, death_KMcurve_t2, composite_KMcurve_t2):

    # Main result:
    # Hospitalization pooled
    # Trials 1-3
    # Death pooled
    # Trials 1-3

    ### Unadjusted
    # Hospitalization pooled
    # Trials 1-3
    # Death pooled
    # Trials 1-3

    # Composite
    # Pooled
    # Trials 1-3

    df1 = hosp_KMcurve_t2.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Hospitalization')).withColumn('category', lit('Trial 2')).withColumn('rank', lit(3))
    df2 = death_KMcurve_t2.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Mortality')).withColumn('category', lit('Trial 2')).withColumn('rank', lit(7))
    df3 = composite_KMcurve_t2.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Composite')).withColumn('category', lit('Trial 2')).withColumn('rank', lit(11))
    

    final = df1.union(df2).union(df3)

    return final
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.ad17d708-d602-47e5-9cd3-0e14d644e11e"),
    composite_KMcurve_t3=Input(rid="ri.vector.main.execute.167f9c74-6263-44f2-bc9a-e13d97a9898c"),
    death_KMcurve_t3=Input(rid="ri.vector.main.execute.f271a6b0-2bad-4027-8bd8-67ba1f6d644e")
)
def Revision_Trial3(hosp_KMcurve_t3, death_KMcurve_t3, composite_KMcurve_t3):

    # Main result:
    # Hospitalization pooled
    # Trials 1-3
    # Death pooled
    # Trials 1-3

    ### Unadjusted
    # Hospitalization pooled
    # Trials 1-3
    # Death pooled
    # Trials 1-3

    # Composite
    # Pooled
    # Trials 1-3

    df1 = hosp_KMcurve_t3.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Hospitalization')).withColumn('category', lit('Trial 3')).withColumn('rank', lit(4))
    df2 = death_KMcurve_t3.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Mortality')).withColumn('category', lit('Trial 3')).withColumn('rank', lit(8))
    df3 = composite_KMcurve_t3.select('risk_ratio','risk_ratio_ll','risk_ratio_ul','risk_reduction','risk_reduction_ll','risk_reduction_ul').where(expr('timeline = 28')).withColumn('Analysis', lit('Main')).withColumn('Outcome', lit('Composite')).withColumn('category', lit('Trial 3')).withColumn('rank', lit(12))
    

    final = df1.union(df2).union(df3)

    return final
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.56cb6112-a291-4e93-817a-fade9d269a4c"),
    Analysis_dataset_merged=Input(rid="ri.foundry.main.dataset.ed08ac9d-3464-48fa-bb22-ce423259bbeb"),
    death_KMcurve_main=Input(rid="ri.vector.main.execute.a0e75935-7689-4229-9050-2286d8523139")
)
def composite_KM_prep(Analysis_dataset_merged, death_KMcurve_main):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.withColumn('event90', expr('CASE WHEN event90 >= 1 THEN 1 ELSE event90 END'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'event90'
    time_to_outcome = 'time90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        # regParam = regparam, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # Calculate MMWS (ATE version)
        # Fit and transform the quantile cutter in Pyspark
        output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # 2. Second, calculate the proportion treated (or control) in each strata
        output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        ###############################################

        # Append the bootstrapped df to our list
        Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # We need to use pandas for KM
        output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # Calculate MMWS (ATE version)
    # Fit and transform the quantile cutter in Pyspark
    output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # 2. Second, calculate the proportion treated (or control) in each strata
    output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

    ###############################################

    # Append the bootstrapped df to our list
    Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # We need to use pandas for KM
    output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.9098de99-ac15-4b57-a8af-0ad0edf2eadf"),
    composite_KMcurve_t2=Input(rid="ri.vector.main.execute.72f99aa7-04c3-4736-8523-addb99dfee0a")
)
def composite_KM_prep_t3(Analysis_dataset_merged, composite_KMcurve_t2):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.withColumn('event90', expr('CASE WHEN event90 >= 1 THEN 1 ELSE event90 END')).where(expr('trial = 3'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'event90'
    time_to_outcome = 'time90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        regParam = 0.0001, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    # Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # NEW: get a pandas data frame of person_id and treatment so we can do stratified sampling
    from sklearn.utils import resample
    unique_persons_df = df_best.select('person_id','treatment', outcome).distinct().toPandas()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # # First - sample some IDS
        random.seed(a = i)
        # sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        ## NEW: Because the data frame has a very SMALL number of treated patients, we will do stratified sampling
        # First, perform a stratified sample of IDs; second convert it to spark data frame for merging; 
        sample_ids_df = resample(unique_persons_df, stratify = unique_persons_df[outcome])
        sample_ids_df = spark.createDataFrame(sample_ids_df[['person_id']])

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # # Calculate MMWS (ATE version)
        # # Fit and transform the quantile cutter in Pyspark
        # output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        # output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        # output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        # output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # # 2. Second, calculate the proportion treated (or control) in each strata
        # output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        # output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        # output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        # Calculate the proportion treated overall
        output_df = output_df.toPandas()
        output_df['strata'] = pd.qcut(output_df['logit'], q = strata_number, labels = False, duplicates = 'drop')
        output_df['strata'] = output_df['strata']+1
        output_df['treated_proportion'] = output_df.groupby(treatment)[treatment].transform('count') / output_df[treatment].count()

        # Calculate the proportion treated in each strata
        output_df['treated_in_strata'] = output_df.groupby(['strata', treatment])[treatment].transform('count') / output_df.groupby(['strata'])['strata'].transform('count')

        # Calculate the MMWS; reweight the proportion treated in strata to the proportion treated
        output_df['MMWS'] = output_df['treated_proportion'] / output_df['treated_in_strata']
        print(output_df[['MMWS', 'propensity', 'treatment']].head())

        ###############################################

        # # Append the bootstrapped df to our list
        # Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # # We need to use pandas for KM
        # output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    # ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    # final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # # Calculate MMWS (ATE version)
    # # Fit and transform the quantile cutter in Pyspark
    # output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    # output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    # output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    # output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # # 2. Second, calculate the proportion treated (or control) in each strata
    # output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    # output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    # output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))
    output_df = output_df.toPandas()
    output_df['strata'] = pd.qcut(output_df['logit'], q = strata_number, labels = False, duplicates = 'drop')
    output_df['strata'] = output_df['strata']+1
    output_df['treated_proportion'] = output_df.groupby(treatment)[treatment].transform('count') / output_df[treatment].count()

    # Calculate the proportion treated in each strata
    output_df['treated_in_strata'] = output_df.groupby(['strata', treatment])[treatment].transform('count') / output_df.groupby(['strata'])['strata'].transform('count')

    # Calculate the MMWS; reweight the proportion treated in strata to the proportion treated
    output_df['MMWS'] = output_df['treated_proportion'] / output_df['treated_in_strata']

    ###############################################

    # # Append the bootstrapped df to our list
    # Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # # We need to use pandas for KM
    # output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.714b59ad-c8af-4711-b3e5-995cee85821f"),
    death_KMcurve_t3=Input(rid="ri.vector.main.execute.f271a6b0-2bad-4027-8bd8-67ba1f6d644e")
)
def composite_KM_t1(Analysis_dataset_merged, death_KMcurve_t3):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.withColumn('event90', expr('CASE WHEN event90 >= 1 THEN 1 ELSE event90 END')).where(expr('trial = 1'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'event90'
    time_to_outcome = 'time90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        # regParam = regparam, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # Calculate MMWS (ATE version)
        # Fit and transform the quantile cutter in Pyspark
        output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # 2. Second, calculate the proportion treated (or control) in each strata
        output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        ###############################################

        # Append the bootstrapped df to our list
        Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # We need to use pandas for KM
        output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # Calculate MMWS (ATE version)
    # Fit and transform the quantile cutter in Pyspark
    output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # 2. Second, calculate the proportion treated (or control) in each strata
    output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

    ###############################################

    # Append the bootstrapped df to our list
    Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # We need to use pandas for KM
    output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.42c0a903-42b7-48a7-9daf-b849c13de72d"),
    composite_KMcurve_t1=Input(rid="ri.vector.main.execute.6d61d07a-5e2c-430a-b3be-49de6c620684")
)
def composite_KM_t2(Analysis_dataset_merged, composite_KMcurve_t1):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.withColumn('event90', expr('CASE WHEN event90 >= 1 THEN 1 ELSE event90 END')).where(expr('trial = 2'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'event90'
    time_to_outcome = 'time90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        # regParam = regparam, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # Calculate MMWS (ATE version)
        # Fit and transform the quantile cutter in Pyspark
        output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # 2. Second, calculate the proportion treated (or control) in each strata
        output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        ###############################################

        # Append the bootstrapped df to our list
        Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # We need to use pandas for KM
        output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # Calculate MMWS (ATE version)
    # Fit and transform the quantile cutter in Pyspark
    output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # 2. Second, calculate the proportion treated (or control) in each strata
    output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

    ###############################################

    # Append the bootstrapped df to our list
    Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # We need to use pandas for KM
    output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.f26742b9-4902-4790-8303-fc2f5f9cdbda"),
    composite_KM_prep=Input(rid="ri.vector.main.execute.56cb6112-a291-4e93-817a-fade9d269a4c")
)
def composite_KMcurve_main( composite_KM_prep):
    
    main_df = composite_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.6d61d07a-5e2c-430a-b3be-49de6c620684"),
    composite_KM_t1=Input(rid="ri.vector.main.execute.714b59ad-c8af-4711-b3e5-995cee85821f")
)
def composite_KMcurve_t1( composite_KM_t1):
    composite_KM_prep = composite_KM_t1
    
    main_df = composite_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    ax.set_title('Composite (Death or Hospitalization)', fontsize=11)
    ax.set_ylabel('Cumulative Incidence (%)', fontsize=10)
    ax.set_ylabel('Cumulative Incidence (%)')
    ax.set_xlabel('Day')
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.72f99aa7-04c3-4736-8523-addb99dfee0a"),
    composite_KM_t2=Input(rid="ri.vector.main.execute.42c0a903-42b7-48a7-9daf-b849c13de72d")
)
def composite_KMcurve_t2( composite_KM_t2):
    composite_KM_prep = composite_KM_t2
    
    main_df = composite_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.167f9c74-6263-44f2-bc9a-e13d97a9898c"),
    composite_KM_prep_t3=Input(rid="ri.vector.main.execute.9098de99-ac15-4b57-a8af-0ad0edf2eadf")
)
def composite_KMcurve_t3( composite_KM_prep_t3):
    composite_KM_prep = composite_KM_prep_t3
    
    main_df = composite_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.6394d3d1-d45f-4e9b-b922-b3d3cba9d884"),
    Analysis_dataset_merged=Input(rid="ri.foundry.main.dataset.ed08ac9d-3464-48fa-bb22-ce423259bbeb"),
    hosp_KMcurve_main=Input(rid="ri.vector.main.execute.374213bc-f702-4f4b-ac08-51cd73e74ff9")
)
def death_KM_prep(Analysis_dataset_merged, hosp_KMcurve_main):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'death90'
    time_to_outcome = 'time_to_death_trunc90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        # regParam = regparam, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # Calculate MMWS (ATE version)
        # Fit and transform the quantile cutter in Pyspark
        output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # 2. Second, calculate the proportion treated (or control) in each strata
        output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        ###############################################

        # Append the bootstrapped df to our list
        Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # We need to use pandas for KM
        output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # Calculate MMWS (ATE version)
    # Fit and transform the quantile cutter in Pyspark
    output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # 2. Second, calculate the proportion treated (or control) in each strata
    output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

    ###############################################

    # Append the bootstrapped df to our list
    Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # We need to use pandas for KM
    output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.fed76e52-21ef-412c-81c6-21144b152a69")
)
def death_KM_prep_t1(Analysis_dataset_merged, hosp_KMcurve_t3):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.where(expr('trial=1'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'death90'
    time_to_outcome = 'time_to_death_trunc90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        # regParam = regparam, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # Calculate MMWS (ATE version)
        # Fit and transform the quantile cutter in Pyspark
        output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # 2. Second, calculate the proportion treated (or control) in each strata
        output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        ###############################################

        # Append the bootstrapped df to our list
        Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # We need to use pandas for KM
        output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # Calculate MMWS (ATE version)
    # Fit and transform the quantile cutter in Pyspark
    output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # 2. Second, calculate the proportion treated (or control) in each strata
    output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

    ###############################################

    # Append the bootstrapped df to our list
    Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # We need to use pandas for KM
    output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.60e9d965-cfd3-48f9-8bc1-a837206fbf20"),
    death_KMcurve_t1=Input(rid="ri.vector.main.execute.8706a7b5-75e3-4f31-ad22-704990ca143d")
)
def death_KM_prep_t2(Analysis_dataset_merged, death_KMcurve_t1):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.where(expr('trial = 2'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'death90'
    time_to_outcome = 'time_to_death_trunc90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        # regParam = regparam, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # Calculate MMWS (ATE version)
        # Fit and transform the quantile cutter in Pyspark
        output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # 2. Second, calculate the proportion treated (or control) in each strata
        output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        ###############################################

        # Append the bootstrapped df to our list
        Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # We need to use pandas for KM
        output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # Calculate MMWS (ATE version)
    # Fit and transform the quantile cutter in Pyspark
    output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # 2. Second, calculate the proportion treated (or control) in each strata
    output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

    ###############################################

    # Append the bootstrapped df to our list
    Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # We need to use pandas for KM
    output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.0d8e70c5-fe7a-441e-974f-cd04576a2d80"),
    death_KMcurve_t2=Input(rid="ri.vector.main.execute.1e9edb45-8d80-4b0b-84b1-679c491536c2")
)
def death_KM_prep_t3(Analysis_dataset_merged, death_KMcurve_t2):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.where(expr('trial = 3'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'death90'
    time_to_outcome = 'time_to_death_trunc90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        regParam = 0.0001, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    # Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # NEW: get a pandas data frame of person_id and treatment so we can do stratified sampling
    from sklearn.utils import resample
    unique_persons_df = df_best.select('person_id','treatment', outcome).distinct().toPandas()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # # First - sample some IDS
        random.seed(a = i)
        # sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        ## NEW: Because the data frame has a very SMALL number of treated patients, we will do stratified sampling
        # First, perform a stratified sample of IDs; second convert it to spark data frame for merging; 
        sample_ids_df = resample(unique_persons_df, stratify = unique_persons_df[outcome])
        sample_ids_df = spark.createDataFrame(sample_ids_df[['person_id']])

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # # Calculate MMWS (ATE version)
        # # Fit and transform the quantile cutter in Pyspark
        # output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        # output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        # output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        # output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # # 2. Second, calculate the proportion treated (or control) in each strata
        # output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        # output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        # output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        # Calculate the proportion treated overall
        output_df = output_df.toPandas()
        output_df['strata'] = pd.qcut(output_df['logit'], q = strata_number, labels = False, duplicates = 'drop')
        output_df['strata'] = output_df['strata']+1
        output_df['treated_proportion'] = output_df.groupby(treatment)[treatment].transform('count') / output_df[treatment].count()

        # Calculate the proportion treated in each strata
        output_df['treated_in_strata'] = output_df.groupby(['strata', treatment])[treatment].transform('count') / output_df.groupby(['strata'])['strata'].transform('count')

        # Calculate the MMWS; reweight the proportion treated in strata to the proportion treated
        output_df['MMWS'] = output_df['treated_proportion'] / output_df['treated_in_strata']
        print(output_df[['MMWS', 'propensity', 'treatment']].head())

        ###############################################

        # # Append the bootstrapped df to our list
        # Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # # We need to use pandas for KM
        # output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    # ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    # final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # # Calculate MMWS (ATE version)
    # # Fit and transform the quantile cutter in Pyspark
    # output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    # output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    # output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    # output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # # 2. Second, calculate the proportion treated (or control) in each strata
    # output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    # output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    # output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))
    output_df = output_df.toPandas()
    output_df['strata'] = pd.qcut(output_df['logit'], q = strata_number, labels = False, duplicates = 'drop')
    output_df['strata'] = output_df['strata']+1
    output_df['treated_proportion'] = output_df.groupby(treatment)[treatment].transform('count') / output_df[treatment].count()

    # Calculate the proportion treated in each strata
    output_df['treated_in_strata'] = output_df.groupby(['strata', treatment])[treatment].transform('count') / output_df.groupby(['strata'])['strata'].transform('count')

    # Calculate the MMWS; reweight the proportion treated in strata to the proportion treated
    output_df['MMWS'] = output_df['treated_proportion'] / output_df['treated_in_strata']

    ###############################################

    # # Append the bootstrapped df to our list
    # Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # # We need to use pandas for KM
    # output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.a0e75935-7689-4229-9050-2286d8523139"),
    death_KM_prep=Input(rid="ri.vector.main.execute.6394d3d1-d45f-4e9b-b922-b3d3cba9d884")
)
def death_KMcurve_main( death_KM_prep):
    
    main_df = death_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.8706a7b5-75e3-4f31-ad22-704990ca143d"),
    death_KM_prep_t1=Input(rid="ri.vector.main.execute.fed76e52-21ef-412c-81c6-21144b152a69")
)
def death_KMcurve_t1( death_KM_prep_t1):
    death_KM_prep = death_KM_prep_t1
    
    main_df = death_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.1e9edb45-8d80-4b0b-84b1-679c491536c2"),
    death_KM_prep_t2=Input(rid="ri.vector.main.execute.60e9d965-cfd3-48f9-8bc1-a837206fbf20")
)
def death_KMcurve_t2( death_KM_prep_t2):
    death_KM_prep = death_KM_prep_t2
    
    main_df = death_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.f271a6b0-2bad-4027-8bd8-67ba1f6d644e"),
    death_KM_prep_t3=Input(rid="ri.vector.main.execute.0d8e70c5-fe7a-441e-974f-cd04576a2d80")
)
def death_KMcurve_t3( death_KM_prep_t3):
    death_KM_prep = death_KM_prep_t3
    
    main_df = death_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.374213bc-f702-4f4b-ac08-51cd73e74ff9"),
    hospital_KM_prep=Input(rid="ri.vector.main.execute.d01cff94-6fe9-40d4-9b4d-8070f433df9e")
)
def hosp_KMcurve_main( hospital_KM_prep):
    
    main_df = hospital_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.0bda773b-7b18-4b66-931a-e31de4e0bed7"),
    hospital_KM_prep_t1=Input(rid="ri.vector.main.execute.ed37b04c-887b-481c-bf01-8e6b323089fc")
)
def hosp_KMcurve_t1( hospital_KM_prep_t1):
    hospital_KM_prep = hospital_KM_prep_t1
    
    main_df = hospital_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.319492af-ca35-4b4b-adfe-6de444eb0aab"),
    hospital_KM_prep_t2=Input(rid="ri.vector.main.execute.e4abd460-4be4-489f-a6cc-e87f5c53f902")
)
def hosp_KMcurve_t2( hospital_KM_prep_t2):
    hospital_KM_prep = hospital_KM_prep_t2
    
    main_df = hospital_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.8c662a24-8221-477e-9199-46d0f13ec63a"),
    hospital_KM_prep_t3=Input(rid="ri.vector.main.execute.a83aa93b-cb90-4f8d-b6e9-45d134af0856")
)
def hosp_KMcurve_t3( hospital_KM_prep_t3):
    hospital_KM_prep = hospital_KM_prep_t3
    
    main_df = hospital_KM_prep
    
    # Right now we have 500 bootstrap survival curves
    # ("time","treatment","control","bootstrap")
    def lower_quantile(series):
        result = series.quantile(0.025)
        return result

    def upper_quantile(series):
        result = series.quantile(0.975)
        return result

    # We have to stack the data frames separately for treatment and control
    df = main_df.where(col('bootstrap') != 999).toPandas()
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
    df_overall = main_df.where(col('bootstrap') == 999).toPandas()
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
    df = df.query('bootstrap != 999') #### WE NEED TO ADD THIS
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
    Output(rid="ri.vector.main.execute.d01cff94-6fe9-40d4-9b4d-8070f433df9e"),
    Analysis_dataset_merged=Input(rid="ri.foundry.main.dataset.ed08ac9d-3464-48fa-bb22-ce423259bbeb")
)
def hospital_KM_prep(Analysis_dataset_merged):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'hospitalized90'
    time_to_outcome = 'time_to_hospitalized_trunc90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        # regParam = regparam, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # Calculate MMWS (ATE version)
        # Fit and transform the quantile cutter in Pyspark
        output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # 2. Second, calculate the proportion treated (or control) in each strata
        output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        ###############################################

        # Append the bootstrapped df to our list
        Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # We need to use pandas for KM
        output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # Calculate MMWS (ATE version)
    # Fit and transform the quantile cutter in Pyspark
    output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # 2. Second, calculate the proportion treated (or control) in each strata
    output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

    ###############################################

    # Append the bootstrapped df to our list
    Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # We need to use pandas for KM
    output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.ed37b04c-887b-481c-bf01-8e6b323089fc"),
    Analysis_dataset_merged=Input(rid="ri.foundry.main.dataset.ed08ac9d-3464-48fa-bb22-ce423259bbeb"),
    composite_KMcurve_main=Input(rid="ri.vector.main.execute.f26742b9-4902-4790-8303-fc2f5f9cdbda")
)
def hospital_KM_prep_t1(Analysis_dataset_merged, composite_KMcurve_main):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.where(expr('trial = 1'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'hospitalized90'
    time_to_outcome = 'time_to_hospitalized_trunc90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        # regParam = regparam, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # Calculate MMWS (ATE version)
        # Fit and transform the quantile cutter in Pyspark
        output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # 2. Second, calculate the proportion treated (or control) in each strata
        output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        ###############################################

        # Append the bootstrapped df to our list
        Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # We need to use pandas for KM
        output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # Calculate MMWS (ATE version)
    # Fit and transform the quantile cutter in Pyspark
    output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # 2. Second, calculate the proportion treated (or control) in each strata
    output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

    ###############################################

    # Append the bootstrapped df to our list
    Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # We need to use pandas for KM
    output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.e4abd460-4be4-489f-a6cc-e87f5c53f902"),
    Analysis_dataset_merged=Input(rid="ri.foundry.main.dataset.ed08ac9d-3464-48fa-bb22-ce423259bbeb"),
    hosp_KMcurve_t1=Input(rid="ri.vector.main.execute.0bda773b-7b18-4b66-931a-e31de4e0bed7")
)
def hospital_KM_prep_t2(Analysis_dataset_merged, hosp_KMcurve_t1):
    hosp_KMcurve_t2 = hosp_KMcurve_t1

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.where(expr('trial = 2'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'hospitalized90'
    time_to_outcome = 'time_to_hospitalized_trunc90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        # regParam = regparam, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # First - sample some IDS
        random.seed(a = i)
        sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # Calculate MMWS (ATE version)
        # Fit and transform the quantile cutter in Pyspark
        output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # 2. Second, calculate the proportion treated (or control) in each strata
        output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        ###############################################

        # Append the bootstrapped df to our list
        Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # We need to use pandas for KM
        output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # Calculate MMWS (ATE version)
    # Fit and transform the quantile cutter in Pyspark
    output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # 2. Second, calculate the proportion treated (or control) in each strata
    output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

    ###############################################

    # Append the bootstrapped df to our list
    Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # We need to use pandas for KM
    output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.a83aa93b-cb90-4f8d-b6e9-45d134af0856"),
    Analysis_dataset_merged=Input(rid="ri.foundry.main.dataset.ed08ac9d-3464-48fa-bb22-ce423259bbeb"),
    hosp_KMcurve_t2=Input(rid="ri.vector.main.execute.319492af-ca35-4b4b-adfe-6de444eb0aab")
)
def hospital_KM_prep_t3(Analysis_dataset_merged, hosp_KMcurve_t2):

    from lifelines import KaplanMeierFitter

    # Set up parameters
    estimand = 'ATE'
    weight_type = 'MMWS'
    bootstraps = 300

    # This node will bootstrap; fit the propensity model on it; calculate the weights, fit the KM curve
    df_best = Analysis_dataset_merged.where(expr('trial = 3'))

    # Set up critical variables
    treatment = 'treatment'
    outcome = 'hospitalized90'
    time_to_outcome = 'time_to_hospitalized_trunc90'

    # Number of strata
    strata_number = 50

    # Make a list of the columns we do not need for propensity modelling
    essential_columns = [
        'person_id',
        'event90',
        'trial',
        'time90',
        'time_to_hospitalized_trunc90',
        'treatment',
        'hospitalized90',
        'death90',
        'time_to_death_trunc90'
        ]

    weight_columns = ['IPTW',
        'MMWS',
        'SW',
        'logit',]

    predictors = [column for column in df_best.columns if column not in essential_columns]

    # Set up the logistic model
    logistic_regression = LogisticRegression(featuresCol = 'predictors', 
        labelCol = treatment, 
        family = 'binomial', 
        maxIter = 1000, 
        elasticNetParam = 0, # This is equivalent to L2
        # fitIntercept = False,
        regParam = 0.0001, # This is 1/C (or alpha)
        # weightCol = 'SW'
        )

    

    ############## NOW - GET BOOTSTRAPS - FIT THE LR MODEL IN EACH; CALCULATE THE WEIGHTS, FIT THE KM FUNCTION, APPEND TO LIST #######

    # # In case we fit KM in a separate step, set up empty list to hold each of the bootstrapped DFs after weighting
    # Output_Prediction_DataFrames = []

    # Create an empty list to store the survival curve data frames for each bootstrap
    CIF_DF_LIST = []

    # 1. First get the complete list of patients 
    unique_persons = df_best.select('person_id').distinct()
    n_unique_persons = unique_persons.count()

    # NEW: get a pandas data frame of person_id and treatment so we can do stratified sampling
    from sklearn.utils import resample
    unique_persons_df = df_best.select('person_id','treatment', outcome).distinct().toPandas()

    # Now for each bootstrap, sample the person_ids (not the rows) 
    for i in np.arange(0, bootstraps):
        
        print('bootstrap location:', i)
        ####### A. BOOTSTRAP SAMPLE ###############
        # # First - sample some IDS
        random.seed(a = i)
        # sample_ids_df = unique_persons.sample(fraction=1.0, seed=i, withReplacement=True)

        ## NEW: Because the data frame has a very SMALL number of treated patients, we will do stratified sampling
        # First, perform a stratified sample of IDs; second convert it to spark data frame for merging; 
        sample_ids_df = resample(unique_persons_df, stratify = unique_persons_df[outcome])
        sample_ids_df = spark.createDataFrame(sample_ids_df[['person_id']])

        # Now merge to the main data frame, df_best; this is our bootstrapped data frame
        cr_sample = sample_ids_df.join(df_best, on = 'person_id', how = 'inner')
        
        # # Set up and fit the propensity model
        # We need to set up a vector assembler in order to use; we input the list of features, and we give that list a name (outputcol)
        assembler = VectorAssembler(inputCols = predictors, outputCol = 'predictors')

        # Set up the logistic regression dataset, transforming our bootstrapped data frame
        logistic_regression_data = assembler.transform(cr_sample)

        ####### B. FIT PROPENSITY MODEL ########################
        # Fit the model to the input data
        model = logistic_regression.fit(logistic_regression_data)

        # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
        output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

        # Get the logit in case we want to use MMWS
        output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

        ####### 3. CALCULATE WEIGHTS ###################################
        if estimand == 'ATE':
            # Modify propensity score for the controls to be 1-propensity
            output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

            # Calculate the inverse weights
            output_df = output_df.withColumn('IPTW', expr('1/propensity'))

            # Stabilize the weights by using the mean of the propensity score for the group
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
        
        elif estimand == 'ATT': 
            
            # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
            output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

            # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
            output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
            output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

            # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
            output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

        # # Calculate MMWS (ATE version)
        # # Fit and transform the quantile cutter in Pyspark
        # output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
        # output_df = output_df.withColumn('strata', expr('strata + 1'))
        
        # # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
        # output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
        # output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

        # # 2. Second, calculate the proportion treated (or control) in each strata
        # output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
        # output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

        # # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
        # output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))

        # Calculate the proportion treated overall
        output_df = output_df.toPandas()
        output_df['strata'] = pd.qcut(output_df['logit'], q = strata_number, labels = False, duplicates = 'drop')
        output_df['strata'] = output_df['strata']+1
        output_df['treated_proportion'] = output_df.groupby(treatment)[treatment].transform('count') / output_df[treatment].count()

        # Calculate the proportion treated in each strata
        output_df['treated_in_strata'] = output_df.groupby(['strata', treatment])[treatment].transform('count') / output_df.groupby(['strata'])['strata'].transform('count')

        # Calculate the MMWS; reweight the proportion treated in strata to the proportion treated
        output_df['MMWS'] = output_df['treated_proportion'] / output_df['treated_in_strata']
        print(output_df[['MMWS', 'propensity', 'treatment']].head())

        ###############################################

        # # Append the bootstrapped df to our list
        # Output_Prediction_DataFrames.append(output_df)

        ######### FIT THE KM CURVE ###################
        # This is a list to hold the curves of each treatment group
        cumulative_incidence_functions = []

        # # We need to use pandas for KM
        # output_df = output_df.toPandas()

        km = KaplanMeierFitter()
        try:
            for group, group_label in zip([0, 1],['control','treatment']):

                km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
                event_observed = output_df.loc[output_df[treatment] == group, outcome], 
                weights = output_df.loc[output_df[treatment] == group, weight_type],
                label = group_label)

                CIF = km.survival_function_
                cumulative_incidence_functions.append(CIF)

            # Join the cumulative incidences of the groups together (axis=1)
            CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
            CIF_DF['bootstrap'] = i
            print(CIF_DF)
            CIF_DF_LIST.append(CIF_DF)
        except:
            None

    # ########### FINAL STEP ############# MERGE DATA FRAMES ####################
    # # Create a stacked dataset of all the bootstrapped dfs (not KM curves)
    # final_bootstraps = reduce(DataFrame.unionAll, Output_Prediction_DataFrames)

    

    ########### REPEAT FOR THE FULL DATASET AND APPEND
    # Set up the logistic regression dataset, transforming our bootstrapped data frame
    logistic_regression_data = assembler.transform(df_best)

    ####### B. FIT PROPENSITY MODEL ########################
    # Fit the model to the input data
    model = logistic_regression.fit(logistic_regression_data)

    # Get the predicted probabilities as an output dataframe; we need: probability, treatment, outcome, time to outcome
    output_df = model.transform(logistic_regression_data).select( [treatment, outcome, time_to_outcome] + [ith(col('probability'), lit(1)).alias('propensity')] )

    # Get the logit in case we want to use MMWS
    output_df = output_df.withColumn('logit', expr('LOG(propensity / (1 - propensity))'))

    ####### 3. CALCULATE WEIGHTS ###################################
    if estimand == 'ATE':
        # Modify propensity score for the controls to be 1-propensity
        output_df = output_df.withColumn('propensity', expr('CASE WHEN treatment = 0 THEN 1 - propensity ELSE propensity END'))

        # Calculate the inverse weights
        output_df = output_df.withColumn('IPTW', expr('1/propensity'))

        # Stabilize the weights by using the mean of the propensity score for the group
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER(PARTITION BY treatment)'))
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))
    
    elif estimand == 'ATT': 
        
        # The IPTW is 1 for the treated groups, and p / 1-p for the control group. This is weighting by the odds
        output_df = output_df.withColumn('IPTW', expr('CASE WHEN treatment = 1 THEN 1 ELSE propensity/(1-propensity) END'))

        # get the stabilized weight - the stabilizer is the proportion treated (for treated SS) or proportion untreated (for controls, i.e., 1 - proportion treated)
        output_df = output_df.withColumn('stabilizer', expr('AVG(propensity) OVER()'))
        output_df = output_df.withColumn('stabilizer', expr('CASE WHEN treatment = 0 THEN 1 - stabilizer ELSE stabilizer END'))

        # Get the stabilized weight; multiply IPTW (ATT) by the stabilizer
        output_df = output_df.withColumn('SW', expr('IPTW * stabilizer'))

    # # Calculate MMWS (ATE version)
    # # Fit and transform the quantile cutter in Pyspark
    # output_df = QuantileDiscretizer(numBuckets = strata_number, inputCol="logit", outputCol="strata").fit(output_df).transform(output_df)
    # output_df = output_df.withColumn('strata', expr('strata + 1'))
    
    # # 1. First calculate the proportion treated overall (proportion treatment, proportion controls, overall)
    # output_df = output_df.withColumn('treated_by_group', expr('COUNT(treatment) OVER(PARTITION BY treatment)')).withColumn('treated_total', expr('COUNT(treatment) OVER()'))
    # output_df = output_df.withColumn('treated_proportion', expr('treated_by_group/treated_total'))

    # # 2. Second, calculate the proportion treated (or control) in each strata
    # output_df = output_df.withColumn('treated_by_strata', expr('COUNT(treatment) OVER(PARTITION BY strata, treatment)')).withColumn('strata_total', expr('COUNT(strata) OVER(PARTITION BY strata)'))
    # output_df = output_df.withColumn('treated_in_strata', expr('treated_by_strata / strata_total'))

    # # 3. Thid, calculate the MMWS - this reweights the proportion treated in a strata (or proportion control in one's strata) to the proportion treated overall. 
    # output_df = output_df.withColumn('MMWS', expr('treated_proportion / treated_in_strata'))
    output_df = output_df.toPandas()
    output_df['strata'] = pd.qcut(output_df['logit'], q = strata_number, labels = False, duplicates = 'drop')
    output_df['strata'] = output_df['strata']+1
    output_df['treated_proportion'] = output_df.groupby(treatment)[treatment].transform('count') / output_df[treatment].count()

    # Calculate the proportion treated in each strata
    output_df['treated_in_strata'] = output_df.groupby(['strata', treatment])[treatment].transform('count') / output_df.groupby(['strata'])['strata'].transform('count')

    # Calculate the MMWS; reweight the proportion treated in strata to the proportion treated
    output_df['MMWS'] = output_df['treated_proportion'] / output_df['treated_in_strata']

    ###############################################

    # # Append the bootstrapped df to our list
    # Output_Prediction_DataFrames.append(output_df)

    ######### FIT THE KM CURVE ###################
    # This is a list to hold the curves of each treatment group
    cumulative_incidence_functions = []

    # # We need to use pandas for KM
    # output_df = output_df.toPandas()

    km = KaplanMeierFitter()

    for group, group_label in zip([0, 1],['control','treatment']):

        km.fit(output_df.loc[output_df[treatment] == group, time_to_outcome], 
        event_observed = output_df.loc[output_df[treatment] == group, outcome], 
        weights = output_df.loc[output_df[treatment] == group, weight_type],
        label = group_label)

        CIF = km.survival_function_
        cumulative_incidence_functions.append(CIF)

    # Join the cumulative incidences of the groups together (axis=1)
    CIF_DF = pd.concat(cumulative_incidence_functions, axis=1)
    CIF_DF['bootstrap'] = 999
    print(CIF_DF)
    CIF_DF_LIST.append(CIF_DF)

    ################### PREPARE OUTPUT #########################
    # Create stack of KM functions (across all bootstraps AND for the overall function); subtract from 1 to get the cuminc. 
    final = pd.concat(CIF_DF_LIST)
    final['treatment'] = 1 - final['treatment']
    final['control'] = 1 - final['control']

    return final.reset_index()

    

