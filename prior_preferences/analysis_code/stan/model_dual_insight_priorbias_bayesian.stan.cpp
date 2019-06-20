
// DRIFT DIFFUSION MODEL: BAYESIAN w/BIASED PRIOR & BIASED DRIFT & INSIGHT on BIAS

// Modified from code by Bradley Doll (https://github.com/dollbb/estRLParam)
// Modified by Naeun Oh : INCLUDING EFFECT OF INSIGHT ON BIAS DRIFT(CHOICE BIAS)

data {
    int<lower=0> NS; // number of subjects
    int<lower=0> NP; // number of item pairs
    int<lower=0> NT; // number of trials
    int<lower=0, upper=1> correct[NS,NP,NT]; // vector of correct responses (0=wrong, 1=correct)
    int<lower=0, upper=1> feedback[NS,NP,NT]; // vector of feedback boxes (0=wrong answer, 1=correct answer)
    real bid_congruence[NS,NP,NT]; // bid for correct minus bid for incorrect item, out of 3.00 GBP
    real item_popularity[NS,NP,NT]; // proportion of all subjects (social and control) who bid more for the correct item than for the incorrect item
    real rt[NS,NP,NT]; // vector of response times in seconds (from PsychoPy recording, not eyelink)
    real rt_mins[NS]; // list of each subject"s lowest response time (except for subject E16, which had implausibly low RTs, and for whom we"ve substituted the lowest RT from all other subjects)

}

parameters {
	
	// DRIFT INTERCEPTS (threshold a)
	// Hyper-parameters
	real<lower=0, upper=3> threshold_int_mean; // Group mean of threshold intercept
	real<lower=0, upper=1> threshold_int_sd; // Pre-transform group standard deviation
	// Subject-level
	real<lower=0> threshold_int[NS];

	// DRIFT LEARNING WEIGHTS (drift weight omega)
	// Hyper-parameters
	real<lower=-3, upper=10> drift_rate_learning_mean; // Group mean of learning coefficient for drift rate
	real<lower=0, upper=pi()/2> drift_rate_learning_sd_unif; // Group standard deviation
	// Subject-level
	vector[NS] drift_rate_learning_raw;

	// CONGRUENCE WEIGHTS (inverse temperatures)
	// Hyper-parameters
    // inverse temperature of effect of preference congruence on prior
	real<lower=-5, upper=5> cong_weight_prior_mean; // Group mean of preference congruence bias weight
	real<lower=0, upper=pi()/2> cong_weight_prior_sd_unif; // Pre-transform group standard deviation of preference congruence bias weight
    // inverse temperature of effect of preference congruence on choice bias
  	real<lower=-5, upper=5> cong_weight_drift_bias_mean; // Group mean of preference congruence bias weight
	real<lower=0, upper=pi()/2> cong_weight_drift_bias_sd_unif; // Pre-transform group standard deviation of preference congruence bias weight
    
    // inverse temperature of effect of popularity on prior
    real<lower=-5, upper=5> insight_mean; // Group mean of "insight" parameter that changes the prior in addition to the value difference
	real<lower=0, upper=pi()/2> insight_sd_unif; // Group SD of insight parameter
    
    // inverse temperature of effect of popularity on choice bias
    real<lower=-5, upper=5> insight_bias_mean; // Group mean of "insight" parameter that changes the prior in addition to the value difference
	real<lower=0, upper=pi()/2> insight_bias_sd_unif; // Group SD of insight parameter
    
	// Subject-level
	vector[NS] cong_weight_prior_raw;
	vector[NS] cong_weight_drift_bias_raw;
    vector[NS] insight_raw;
    vector[NS] insight_bias_raw;

	    // NON-DECISION TIMES used to transform
//     vector<lower=0>[NS] ndt_raw;
    // NON-DECISION TIMES
	real<lower=0, upper=rt_mins[1]> ndt_1;
	real<lower=0, upper=rt_mins[2]> ndt_2;
	real<lower=0, upper=rt_mins[3]> ndt_3;
	real<lower=0, upper=rt_mins[4]> ndt_4;
	real<lower=0, upper=rt_mins[5]> ndt_5;
	real<lower=0, upper=rt_mins[6]> ndt_6;
	real<lower=0, upper=rt_mins[7]> ndt_7;
	real<lower=0, upper=rt_mins[8]> ndt_8;
	real<lower=0, upper=rt_mins[9]> ndt_9;
	real<lower=0, upper=rt_mins[10]> ndt_10;
	real<lower=0, upper=rt_mins[11]> ndt_11;
	real<lower=0, upper=rt_mins[12]> ndt_12;
	real<lower=0, upper=rt_mins[13]> ndt_13;
	real<lower=0, upper=rt_mins[14]> ndt_14;
	real<lower=0, upper=rt_mins[15]> ndt_15;
	real<lower=0, upper=rt_mins[16]> ndt_16;
	real<lower=0, upper=rt_mins[17]> ndt_17;
	real<lower=0, upper=rt_mins[18]> ndt_18;
	real<lower=0, upper=rt_mins[19]> ndt_19;
	real<lower=0, upper=rt_mins[20]> ndt_20;
	real<lower=0, upper=rt_mins[21]> ndt_21;
	real<lower=0, upper=rt_mins[22]> ndt_22;
	real<lower=0, upper=rt_mins[23]> ndt_23;
	real<lower=0, upper=rt_mins[24]> ndt_24;
	real<lower=0, upper=rt_mins[25]> ndt_25;
	real<lower=0, upper=rt_mins[26]> ndt_26;
	real<lower=0, upper=rt_mins[27]> ndt_27;
	real<lower=0, upper=rt_mins[28]> ndt_28;
	real<lower=0, upper=rt_mins[29]> ndt_29;
	real<lower=0, upper=rt_mins[30]> ndt_30;
	real<lower=0, upper=rt_mins[31]> ndt_31;
}

transformed parameters {
//     vector[NS] ndt = rt_mins + ndt_raw;   // NON-DECISION TIMES with lower and upper bounds
	real drift_rate_learning_sd;
	real<lower=0> cong_weight_prior_sd;
	real<lower=0> cong_weight_drift_bias_sd;
    real<lower=0> insight_sd;
    real<lower=0> insight_bias_sd;
	vector[NS] drift_rate_learning;
	vector[NS] cong_weight_prior;
	vector[NS] cong_weight_drift_bias;
    vector[NS] insight;
    vector[NS] insight_bias;
	real threshold_int_a;
	real threshold_int_b;
	real<lower=0> non_decision_time_int[NS]; 
 
 	// Normally distributed priors
 	drift_rate_learning_sd <- 0 + 0.5*tan(drift_rate_learning_sd_unif);
 	cong_weight_prior_sd <- 0 + 2.5*tan(cong_weight_prior_sd_unif);
 	cong_weight_drift_bias_sd <- 0 + 2.5*tan(cong_weight_drift_bias_sd_unif);
    insight_sd = 0 + 2.5*tan(insight_sd_unif);
    insight_bias_sd <- 0 + 2.5*tan(insight_bias_sd_unif);
    
 	drift_rate_learning <- drift_rate_learning_mean + drift_rate_learning_sd * drift_rate_learning_raw;
 	cong_weight_prior <- cong_weight_prior_mean + cong_weight_prior_sd * cong_weight_prior_raw;
 	cong_weight_drift_bias <- cong_weight_drift_bias_mean + cong_weight_drift_bias_sd * cong_weight_drift_bias_raw;
    insight = insight_mean + insight_sd * insight_raw;
    insight_bias <- insight_bias_mean + insight_bias_sd * insight_bias_raw;

 	// Gamma distributed priors
 	threshold_int_a <- pow((threshold_int_mean / threshold_int_sd),2); // Shape parameter of threshold intercept
 	threshold_int_b <- threshold_int_mean / pow(threshold_int_sd, 2); // Rate (inverse scale) parameter of threshold intercept

    	// Non-decision time vector
 	non_decision_time_int[1] = ndt_1;
 	non_decision_time_int[2] = ndt_2;
 	non_decision_time_int[3] = ndt_3;
 	non_decision_time_int[4] = ndt_4;
 	non_decision_time_int[5] = ndt_5;
 	non_decision_time_int[6] = ndt_6;
 	non_decision_time_int[7] = ndt_7;
 	non_decision_time_int[8] = ndt_8;
 	non_decision_time_int[9] = ndt_9;
 	non_decision_time_int[10] = ndt_10;
 	non_decision_time_int[11] = ndt_11;
 	non_decision_time_int[12] = ndt_12;
 	non_decision_time_int[13] = ndt_13;
 	non_decision_time_int[14] = ndt_14;
 	non_decision_time_int[15] = ndt_15;
 	non_decision_time_int[16] = ndt_16;
 	non_decision_time_int[17] = ndt_17;
 	non_decision_time_int[18] = ndt_18;
 	non_decision_time_int[19] = ndt_19;
 	non_decision_time_int[20] = ndt_20;
 	non_decision_time_int[21] = ndt_21;
 	non_decision_time_int[22] = ndt_22;
 	non_decision_time_int[23] = ndt_23;
 	non_decision_time_int[24] = ndt_24;
 	non_decision_time_int[25] = ndt_25;
 	non_decision_time_int[26] = ndt_26;
 	non_decision_time_int[27] = ndt_27;
 	non_decision_time_int[28] = ndt_28;
 	non_decision_time_int[29] = ndt_29;
 	non_decision_time_int[30] = ndt_30;
 	non_decision_time_int[31] = ndt_31;
}

model {
	vector[2] q; // set up a 2-item array to hold two probability values, one for the correct and one for the incorrect answer
	vector[2] bid_cong_prior; // vector to hold bid congruence to calculate softmax for prior
	vector[2] bid_cong_drift; // and for drift bias

	real pri; // set up variable to hold PRIOR PERCEIVED probability of correct answer being correct
	real pr; // set up variable to hold PERCEIVED probability of correct answer being correct
	real a; // placeholder variable for threshold
	real ti; // non-decision time
	real b; // drift bias
	real v; // drift rate

	real p_correct; // placeholder variable for MODEL'S probability of a correct response (only used in generated quantities)

	// Convenience variables for Bayes rule
    real pr_d_c; // p(data | correct)
    real pr_d_not_c; // p(data | ~correct)
    int cf; // cumulative correct feedback so far for this pair

	// Hyper-parameter priors
	threshold_int_mean ~ normal(1,10);
	threshold_int_sd ~ uniform(0,2);

	drift_rate_learning_mean ~ normal(0,20);
	drift_rate_learning_sd_unif ~ uniform(0,pi()/2);

	cong_weight_prior_mean ~ normal(0, 5);
	cong_weight_prior_sd_unif ~ uniform(0,pi()/2);

	cong_weight_drift_bias_mean ~ normal(0, 5);
	cong_weight_drift_bias_sd_unif ~ uniform(0,pi()/2);
    
    insight_mean ~ normal(0, 5);
	insight_sd_unif ~ uniform(0,pi()/2);
    
    insight_bias_mean ~ normal(0, 5);
	insight_bias_sd_unif ~ uniform(0,pi()/2);

	// Subject-level priors
	threshold_int ~ gamma(threshold_int_a, threshold_int_b);
	drift_rate_learning_raw ~ normal(0, 1);
	cong_weight_prior_raw ~ normal(0, 1);
	cong_weight_drift_bias_raw ~ normal(0, 1);
    insight_raw ~ normal(0, 1);
    insight_bias_raw ~ normal(0, 1);
    non_decision_time_int ~ uniform(0, rt_mins); // Non-hierarchical priors for non-decision times, with the upper bound being the lowest RT for that subject
		// Priors for non-decision time
// 		ndt[s] ~ uniform(0, rt_mins[s]); // Non-hierarchical priors for non-decision times, with the upper bound being the lowest RT for that subject
//print ("threshold_int: ", threshold_int, " fixation_bias: ", fixation_bias, " cong_weight: ", cong_weight, " drift_rate_learning: ", drift_rate_learning);
	
	for (s in 1:NS) {
        a = threshold_int[s];
		ti = non_decision_time_int[s];
		
		for (p in 1:NP) {
            
			cf <- 0;
			bid_cong_prior[1] <- (cong_weight_prior[s] * bid_congruence[s,p,1]) + (insight[s] * (item_popularity[s,p,1] - 0.5));
			bid_cong_prior[2] <- 0;
			bid_cong_drift[1] <- (cong_weight_drift_bias[s] * bid_congruence[s,p,1]) + (insight_bias[s] * (item_popularity[s,p,1] - 0.5));  // included effect of popularity on choice bias
			bid_cong_drift[2] <- 0;
			pri <- softmax(bid_cong_prior)[1]; // prior is a softmax of the bid difference with inverse temperature cong_weight_prior[s]
			b <- softmax(bid_cong_drift)[1]; // drift bias term is a softmax of the bid difference with inverse temperature cong_weight_drift_bias[s]
			q[2] <- pri; // perceived probability of CORRECT answer being correct
			q[1] <- (1 - pri); // perceived probability of WRONG answer being correct

			for (t in 1:NT) {
				v <- drift_rate_learning[s] * ( q[2] - q[1] ) ;

				// if (v == 0) {
				// 	p_correct <- b; 
				// }
				// else {
				// 	p_correct <- 1 - ( (1 - exp(-2*v*a*(1-b)) ) / (exp(2*v*a*b) - exp(-2*v*a*(1-b)) ) );
				// }

				// // Sampling statement for choice data
				// if (is_nan(p_correct) == 1)
				// 	reject("q, pri, softmax, insight: ", q, " ",pri, " ", softmax(bid_cong_prior)[1], " ", insight[s]);
				// correct[s,p,t] ~ bernoulli(p_correct); 

				// Sampling statement for response time data
				if ( rt[s,p,t] > 0.41 ) { // Lowest plausible RT in data (excluding implausible RTs from participant E16)

					if (correct[s,p,t] == 1) {
						rt[s,p,t] ~ wiener(a, ti, b, v); // For correct responses, return upper boundary
					}
					else {
						rt[s,p,t] ~ wiener(a, ti, 1-b, -v); // For incorrect responses, return lower boundary
					}
				}
				cf <- cf + feedback[s,p,t]; // increase cumulative feedback by feedback on this trial
                pr_d_c = exp(binomial_log(cf,t,0.8));
                pr_d_not_c = exp(binomial_log(cf,t,0.2));
                pr <- (pr_d_c * pri) / ( (pr_d_c * pri) + (pr_d_not_c * (1-pri)) ); // Perceived probability of correct answer being correct, given cumulative feedback
				q[2] <- pr;
				q[1] <- (1 - pr);
				
			}
		}
	}
}

generated quantities {
	vector[NS*NP*NT] log_lik_rt; 
	vector[NS*NP*NT] log_lik_resp;
	vector[NS*NP*NT] v_store; // store the "v" drift rate values for each trial and sample
	vector[NS*NP*NT] b_store; // store the "b" starting point values
	int ix;
	vector[2] q; // set up a 2-item array to hold two Q values, one for the correct and one for the incorrect answer
	vector[2] bid_cong_prior; // vector to hold bid congruence to calculate softmax for prior
	vector[2] bid_cong_drift; // and for drift bias

	real pri; // set up variable to hold PRIOR PERCEIVED probability of correct answer being correct
	real pr; // set up variable to hold PERCEIVED probability of correct answer being correct
	real a; // placeholder variable for threshold
	real ti; // non-decision time
	real b; // drift bias
	real v; // drift rate

	real p_correct; // placeholder variable for probability of a correct response

	// Convenience variables for Bayes rule
    real pr_d_c; // p(data | correct)
    real pr_d_not_c; // p(data | ~correct)
    int cf; // cumulative correct feedback so far for this pair

	for (s in 1:NS) {
		a = threshold_int[s];
		ti = non_decision_time_int[s];
        
		for (p in 1:NP) {
			
			cf <- 0;
			bid_cong_prior[1] <- (cong_weight_prior[s] * bid_congruence[s,p,1]) + (insight[s] * (item_popularity[s,p,1] - 0.5));
			bid_cong_prior[2] <- 0;
            bid_cong_drift[1] <- (cong_weight_drift_bias[s] * bid_congruence[s,p,1]) + (insight_bias[s] * (item_popularity[s,p,1] - 0.5));  // included effect of popularity on choice bias
			bid_cong_drift[2] <- 0;
			pri <- softmax(bid_cong_prior)[1]; // prior is a softmax of the bid difference with inverse temperature cong_weight_prior[s]
			b <- softmax(bid_cong_drift)[1]; // drift bias term is a softmax of the bid difference with inverse temperature cong_weight_drift_bias[s]
			q[2] <- pri; // perceived probability of CORRECT answer being correct
			q[1] <- (1 - pri); // perceived probability of WRONG answer being correct

			for (t in 1:NT) {
				ix <- (s-1)*NP*NT + (p-1)*NT + t; // index of log_lik_rt and lik_resp vectors
				
                v <- drift_rate_learning[s] * ( q[2] - q[1] ) ;

				v_store[ix] <- v;
				b_store[ix] <- b;

				if (v != 0) {
                    p_correct = 1 - ( (1 - exp(-2*v*a*(1-b)) ) / (exp(2*v*a*b) - exp(-2*v*a*(1-b)) ) );
				}
				else {
					p_correct = b; 
				}


				if (correct[s,p,t] == 1) {
					if (rt[s,p,t] >  -1) { // If RT data are not missing for that trial
						log_lik_rt[ix] = wiener_lpdf(rt[s,p,t]| a, ti, b, v); // For correct responses, return upper boundary
					}
					log_lik_resp[ix] = log(p_correct); // Log probability of being correct
				}
				else {
					if (rt[s,p,t] > -1) {
						log_lik_rt[ix] = wiener_lpdf(rt[s,p,t]| a, ti, b, v);  // For incorrect responses, return lower boundary
					}
					log_lik_resp[ix] = log(1-p_correct); // Log probability of being wrong
				}
				cf <- cf + feedback[s,p,t]; // increase cumulative feedback by feedback on this trial
                pr_d_c = exp(binomial_lpmf(cf|t,0.8));
                pr_d_not_c = exp(binomial_lpmf(cf|t,0.2));
                pr <- (pr_d_c * pri) / ( (pr_d_c * pri) + (pr_d_not_c * (1-pri)) ); // Perceived probability of correct answer being correct, given cumulative feedback
				q[2] <- pr;
				q[1] <- (1 - pr);
			}
		}
	}
}