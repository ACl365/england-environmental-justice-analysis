# Data Dictionary

This document provides a description of every feature used in the project, including its source, units, type, and how it was derived/normalized.

## Unified Dataset (unified_dataset_with_air_quality.csv)

| Feature | Description | Source | Units | Type | Derivation/Normalization |
|---|---|---|---|---|---|
| lsoa_code | Lower Super Output Area code | ONS | N/A | String | N/A |
| lsoa_name | Lower Super Output Area name | ONS | N/A | String | N/A |
| lad_code | Local Authority District code | ONS | N/A | String | N/A |
| lad_name | Local Authority District name | ONS | N/A | String | N/A |
| imd_rank | Index of Multiple Deprivation rank | DCLG | N/A | Integer | N/A |
| imd_decile | Index of Multiple Deprivation decile | DCLG | N/A | Integer | N/A |
| imd_score_normalized | Normalized Index of Multiple Deprivation score | DCLG | N/A | Float | Normalized to 0-100 scale |
| total_population_mid_2015_excluding_prisoners | Total population (excluding prisoners) | ONS | People | Integer | N/A |
| dependent_children_aged_0_15_mid_2015_excluding_prisoners | Number of dependent children (excluding prisoners) | ONS | People | Integer | N/A |
| population_aged_16_59_mid_2015_excluding_prisoners | Population aged 16-59 (excluding prisoners) | ONS | People | Integer | N/A |
| older_population_aged_60_and_over_mid_2015_excluding_prisoners | Older population aged 60 and over (excluding prisoners) | ONS | People | Integer | N/A |
| working_age_population_18_59/64_for_use_with_employment_deprivation_domain_excluding_prisoners | Working age population | ONS | People | Integer | N/A |
| index_of_multiple_deprivation_imd_score | Index of Multiple Deprivation score | DCLG | N/A | Float | N/A |
| index_of_multiple_deprivation_imd_rank_where_1_is_most_deprived | Index of Multiple Deprivation rank | DCLG | N/A | Integer | N/A |
| index_of_multiple_deprivation_imd_decile_where_1_is_most_deprived_10%_of_lsoas | Index of Multiple Deprivation decile | DCLG | N/A | Integer | N/A |
| income_score_rate | Income score rate | DCLG | N/A | Float | N/A |
| income_rank_where_1_is_most_deprived | Income rank | DCLG | N/A | Integer | N/A |
| income_decile_where_1_is_most_deprived_10%_of_lsoas | Income decile | DCLG | N/A | Integer | N/A |
| employment_score_rate | Employment score rate | DCLG | N/A | Float | N/A |
| employment_rank_where_1_is_most_deprived | Employment rank | DCLG | N/A | Integer | N/A |
| employment_decile_where_1_is_most_deprived_10%_of_lsoas | Employment decile | DCLG | N/A | Integer | N/A |
| education_skills_and_training_score | Education, Skills and Training score | DCLG | N/A | Float | N/A |
| education_skills_and_training_rank_where_1_is_most_deprived | Education, Skills and Training rank | DCLG | N/A | Integer | N/A |
| education_skills_and_training_decile_where_1_is_most_deprived_10%_of_lsoas | Education, Skills and Training decile | DCLG | N/A | Integer | N/A |
| health_deprivation_and_disability_score | Health Deprivation and Disability score | DCLG | N/A | Float | N/A |
| health_deprivation_and_disability_rank_where_1_is_most_deprived | Health Deprivation and Disability rank | DCLG | N/A | Integer | N/A |
| health_deprivation_and_disability_decile_where_1_is_most_deprived_10%_of_lsoas | Health Deprivation and Disability decile | DCLG | N/A | Integer | N/A |
| crime_score | Crime score | DCLG | N/A | Float | N/A |
| crime_rank_where_1_is_most_deprived | Crime rank | DCLG | N/A | Integer | N/A |
| crime_decile_where_1_is_most_deprived_10%_of_lsoas | Crime decile | DCLG | N/A | Integer | N/A |
| barriers_to_housing_and_services_score | Barriers to Housing and Services score | DCLG | N/A | Float | N/A |
| barriers_to_housing_and_services_rank_where_1_is_most_deprived | Barriers to Housing and Services rank | DCLG | N/A | Integer | N/A |
| barriers_to_housing_and_services_decile_where_1_is_most_deprived_10%_of_lsoas | Barriers to Housing and Services decile | DCLG | N/A | Integer | N/A |
| living_environment_score | Living Environment score | DCLG | N/A | Float | N/A |
| living_environment_rank_where_1_is_most_deprived | Living Environment rank | DCLG | N/A | Integer | N/A |
| living_environment_decile_where_1_is_most_deprived_10%_of_lsoas | Living Environment decile | DCLG | N/A | Integer | N/A |
| income_deprivation_affecting_children_index_idaci_score_rate | Income Deprivation Affecting Children Index (IDACI) score rate | DCLG | N/A | Float | N/A |
| income_deprivation_affecting_children_index_idaci_rank_where_1_is_most_deprived | Income Deprivation Affecting Children Index (IDACI) rank | DCLG | N/A | Integer | N/A |
| income_deprivation_affecting_children_index_idaci_decile_where_1_is_most_deprived_10%_of_lsoas | Income Deprivation Affecting Children Index (IDACI) decile | DCLG | N/A | Integer | N/A |
| income_deprivation_affecting_older_people_idaopi_score_rate | Income Deprivation Affecting Older People (IDAOPI) score rate | DCLG | N/A | Float | N/A |
| income_deprivation_affecting_older_people_idaopi_rank_where_1_is_most_deprived | Income Deprivation Affecting Older People (IDAOPI) rank | DCLG | N/A | Integer | N/A |
| income_deprivation_affecting_older_people_idaopi_decile_where_1_is_most_deprived_10%_of_lsoas | Income Deprivation Affecting Older People (IDAOPI) decile | DCLG | N/A | Integer | N/A |
| children_and_young_people_sub_domain_score | Children and Young People sub-domain score | DCLG | N/A | Float | N/A |
| children_and_young_people_sub_domain_rank_where_1_is_most_deprived | Children and Young People sub-domain rank | DCLG | N/A | Integer | N/A |
| children_and_young_people_sub_domain_decile_where_1_is_most_deprived_10%_of_lsoas | Children and Young People sub-domain decile | DCLG | N/A | Integer | N/A |
| adult_skills_sub_domain_score | Adult Skills sub-domain score | DCLG | N/A | Float | N/A |
| adult_skills_sub_domain_rank_where_1_is_most_deprived | Adult Skills sub-domain rank | DCLG | N/A | Integer | N/A |
| adult_skills_sub_domain_decile_where_1_is_most_deprived_10%_of_lsoas | Adult Skills sub-domain decile | DCLG | N/A | Integer | N/A |
| geographical_barriers_sub_domain_score | Geographical Barriers sub-domain score | DCLG | N/A | Float | N/A |
| geographical_barriers_sub_domain_rank_where_1_is_most_deprived | Geographical Barriers sub-domain rank | DCLG | N/A | Integer | N/A |
| geographical_barriers_sub_domain_decile_where_1_is_most_deprived_10%_of_lsoas | Geographical Barriers sub-domain decile | DCLG | N/A | Integer | N/A |
| wider_barriers_sub_domain_score | Wider Barriers sub-domain score | DCLG | N/A | Float | N/A |
| wider_barriers_sub_domain_rank_where_1_is_most_deprived | Wider Barriers sub-domain rank | DCLG | N/A | Integer | N/A |
| wider_barriers_sub_domain_decile_where_1_is_most_deprived_10%_of_lsoas | Wider Barriers sub-domain decile | DCLG | N/A | Integer | N/A |
| indoors_sub_domain_score | Indoors sub-domain score | DCLG | N/A | Float | N/A |
| indoors_sub_domain_rank_where_1_is_most_deprived | Indoors sub-domain rank | DCLG | N/A | Integer | N/A |
| indoors_sub_domain_decile_where_1_is_most_deprived_10%_of_lsoas | Indoors sub-domain decile | DCLG | N/A | Integer | N/A |
| outdoors_sub_domain_score | Outdoors sub-domain score | DCLG | N/A | Float | N/A |
| outdoors_sub_domain_rank_where_1_is_most_deprived | Outdoors sub-domain rank | DCLG | N/A | Integer | N/A |
| outdoors_sub_domain_decile_where_1_is_most_deprived_10%_of_lsoas | Outdoors sub-domain decile | DCLG | N/A | Integer | N/A |
| working_age_population_18_59/64_for_use_with_employment_deprivation_domain_excluding_prisoners_ | Working age population | ONS | People | Integer | N/A |
| data_quality | Data quality flag | DCLG | N/A | Integer | N/A |
| NO2 | Nitrogen Dioxide concentration | DEFRA | ug/m3 | Float | N/A |
| O3 | Ozone concentration | DEFRA | ug/m3 | Float | N/A |
| PM10 | Particulate Matter < 10um concentration | DEFRA | ug/m3 | Float | N/A |
| PM2.5 | Particulate Matter < 2.5um concentration | DEFRA | ug/m3 | Float | N/A |
| PM2.5_normalized | Normalized Particulate Matter < 2.5um concentration | DEFRA | N/A | Float | Normalized to 0-1 scale |
| PM10_normalized | Normalized Particulate Matter < 10um concentration | DEFRA | N/A | Float | Normalized to 0-1 scale |
| NO2_normalized | Normalized Nitrogen Dioxide concentration | DEFRA | N/A | Float | Normalized to 0-1 scale |
| air_pollution_index | Air pollution index | Calculated | N/A | Float | Weighted average of normalized pollutant concentrations: (0.4 * NO2_normalized + 0.3 * PM2.5_normalized + 0.2 * PM10_normalized + 0.1 * (1 - O3/max(O3))), where O3 is inverted as higher O3 at ground level is generally associated with lower NO2. **Rationale:** Weights reflect the relative known health impacts and regulatory focus (higher weight for NO2 and PM2.5). Ozone is included but inverted and weighted lower due to its complex relationship with other pollutants at ground level. Normalization ensures pollutants are comparable. |
| env_justice_index | Environmental justice index | Calculated | N/A | Float | Calculated as (air_pollution_index * imd_score_normalized)^0.5, representing the geometric mean of pollution burden and socioeconomic deprivation. **Rationale:** The geometric mean is used instead of an arithmetic mean to ensure that areas must have *both* high pollution *and* high deprivation to score highly on the index. This specifically targets the 'double burden' concept central to environmental justice. It prevents areas with extremely high pollution but low deprivation (or vice-versa) from dominating the index. |

## Health Indicators Dataset (health_indicators_by_lad.csv)

| Feature | Description | Source | Units | Type | Derivation/Normalization |
|---|---|---|---|---|---|
| local_authority_code | Local Authority District code | NHS OF | N/A | String | N/A |
| chronic_conditions_value | Chronic conditions value | NHS OF | N/A | Float | N/A |
| chronic_conditions_lower_ci | Chronic conditions lower confidence interval | NHS OF | N/A | Float | N/A |
| chronic_conditions_upper_ci | Chronic conditions upper confidence interval | NHS OF | N/A | Float | N/A |
| chronic_conditions_standardised_ratio | Chronic conditions standardised ratio | NHS OF | N/A | Float | N/A |
| chronic_conditions_name | Chronic conditions name | NHS OF | N/A | String | N/A |
| chronic_conditions_description | Chronic conditions description | NHS OF | N/A | String | N/A |
| local_authority_name | Local Authority District name | NHS OF | N/A | String | N/A |
| asthma_diabetes_epilepsy_value | Asthma, diabetes, epilepsy value | NHS OF | N/A | Float | N/A |
| asthma_diabetes_epilepsy_lower_ci | Asthma, diabetes, epilepsy lower confidence interval | NHS OF | N/A | Float | N/A |
| asthma_diabetes_epilepsy_upper_ci | Asthma, diabetes, epilepsy upper confidence interval | NHS OF | N/A | Float | N/A |
| asthma_diabetes_epilepsy_standardised_ratio | Asthma, diabetes, epilepsy standardised ratio | NHS OF | N/A | Float | N/A |
| asthma_diabetes_epilepsy_name | Asthma, diabetes, epilepsy name | NHS OF | N/A | String | N/A |
| asthma_diabetes_epilepsy_description | Asthma, diabetes, epilepsy description | NHS OF | N/A | String | N/A |
| lrti_children_value | Lower respiratory tract infections in children value | NHS OF | N/A | Float | N/A |
| lrti_children_lower_ci | Lower respiratory tract infections in children lower confidence interval | NHS OF | N/A | Float | N/A |
| lrti_children_upper_ci | Lower respiratory tract infections in children upper confidence interval | NHS OF | N/A | Float | N/A |
| lrti_children_standardised_ratio | Lower respiratory tract infections in children standardised ratio | NHS OF | N/A | Float | N/A |
| lrti_children_name | Lower respiratory tract infections in children name | NHS OF | N/A | String | N/A |
| lrti_children_description | Lower respiratory tract infections in children description | NHS OF | N/A | String | N/A |
| acute_conditions_value | Acute conditions value | NHS OF | N/A | Float | N/A |
| acute_conditions_lower_ci | Acute conditions lower confidence interval | NHS OF | N/A | Float | N/A |
| acute_conditions_upper_ci | Acute conditions upper confidence interval | NHS OF | N/A | Float | N/A |
| acute_conditions_standardised_ratio | Acute conditions standardised ratio | NHS OF | N/A | Float | N/A |
| acute_conditions_name | Acute conditions name | NHS OF | N/A | String | N/A |
| acute_conditions_description | Acute conditions description | NHS OF | N/A | String | N/A |
| chronic_conditions_normalized | Normalized chronic conditions value | NHS OF | N/A | Float | Normalized to 0-1 scale |
| asthma_diabetes_epilepsy_normalized | Normalized asthma, diabetes, epilepsy value | NHS OF | N/A | Float | Normalized to 0-1 scale |
| lrti_children_normalized | Normalized lower respiratory tract infections in children value | NHS OF | N/A | Float | Normalized to 0-1 scale |
| acute_conditions_normalized | Normalized acute conditions value | NHS OF | N/A | Float | Normalized to 0-1 scale |
| respiratory_health_index | Respiratory health index | Calculated | N/A | Float | Weighted average of normalized respiratory health indicators: (0.35 * chronic_conditions_normalized + 0.35 * asthma_diabetes_epilepsy_normalized + 0.15 * lrti_children_normalized + 0.15 * acute_conditions_normalized). Higher values indicate better respiratory health outcomes. **Rationale:** Weights are assigned based on the severity and prevalence reflected in the source indicators, giving slightly higher importance to chronic conditions and conditions like asthma often linked to air quality. LRTI in children and acute conditions are included as important but potentially less chronic indicators. Normalization allows combining different scales. The index is constructed so higher values = better health for intuitive interpretation. |
| overall_health_index | Overall health index | Calculated | N/A | Float | Comprehensive health metric combining respiratory_health_index (60% weight) with other health indicators including standardized mortality ratios and hospital admission rates (40% weight). Higher values indicate better overall health outcomes. **Rationale:** Provides a broader health context, weighting the specific respiratory focus alongside general mortality and morbidity indicators available at the LAD level. |