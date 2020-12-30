import pandas as pd
import numpy as np
from scipy.stats import norm


def lineup_predictions_as_array(df, targets):
    return df.pipe(
        lambda y: np.stack(
            y.groupby("date").apply(
                lambda z: 
                    np.stack(
                        [
                            z[[f"{target}_prediction", f"{target}_uncertainty"]].values
                            for target in targets
                        ],
                        axis=1
                    )
                )
            )
        )


def lineup_eff_predictions_as_array(df, targets):
    return df.pipe(
        lambda y: np.stack(
            y.groupby("date").apply(
                lambda z: 
                    np.stack(
                        [
                            z[[f"{target}_pct_prediction", f"{target}_pct_uncertainty", f"{target}a_prediction"]].values
                            for target in targets
                        ],
                        axis=1
                    )
                )
            )
        )


def lineup_stats_as_array(df, targets):
    return df.pipe(
        lambda y: np.stack(
            y.groupby("date").apply(lambda z: z[targets].values)
            )
        )


def lineup_eff_stats_as_array(df, targets):
    return df.pipe(
        lambda y: np.stack(
            y.groupby("date").apply(
                lambda z: 
                    np.stack(
                        [
                            z[[f"{target}_pct", f"{target}a"]].values
                            for target in targets
                        ],
                        axis=1
                    )
                )
            )
        )


def expected_sum_categories_won(lineups, opp_lineups):
    diff = np.sum(opp_lineups[:,:,:,0], axis=1) - np.sum(lineups[:,:,:,0], axis=1)
    var = np.sqrt(np.sum(lineups[:,:,:,1], axis=1) + np.sum(opp_lineups[:,:,:,1], axis=1))
    z = diff / var
    p = 1 - norm.cdf(z)
    return p


def expected_efficiency_categories_won(lineups, opp_lineups):
    perc_fta = lineups[:,:,:,2] / np.sum(lineups[:,:,:,2], axis=1, keepdims=True)
    opp_perc_fta = opp_lineups[:,:,:,2] / np.sum(opp_lineups[:,:,:,2], axis=1, keepdims=True)
    diff = np.sum(opp_lineups[:,:,:,0] * opp_perc_fta, axis=1)  - np.sum(lineups[:,:,:,0] * perc_fta, axis=1)
    var = np.sqrt(np.sum(lineups[:,:,:,1] * perc_fta**2, axis=1) + np.sum(opp_lineups[:,:,:,1] * opp_perc_fta**2, axis=1))
    z = diff / var
    p = 1 - norm.cdf(z)
    return p


def actual_sum_categories_won(lineups, opp_lineups):
    return (np.nansum(lineups, axis=1) > np.nansum(opp_lineups, axis=1)).astype(int)


def actual_efficiency_categories_won(lineups, opp_lineups):
    perc_fta = lineups[:,:,:,1] / np.nansum(lineups[:,:,:,1], axis=1, keepdims=True)
    opp_perc_fta = opp_lineups[:,:,:,1] / np.nansum(opp_lineups[:,:,:,1], axis=1, keepdims=True)
    return (
        np.nansum(lineups[:,:,:,0] * perc_fta, axis=1)
        > np.nansum(opp_lineups[:,:,:,0] * opp_perc_fta, axis=1)
    ).astype(int)


def expected_categories_won(lineups_df, opp_lineups_df, sum_targets, eff_targets):
    lineups = lineup_predictions_as_array(lineups_df, sum_targets)
    opp_lineups = lineup_predictions_as_array(opp_lineups_df, sum_targets)

    lineups_eff = lineup_eff_predictions_as_array(lineups_df, eff_targets)
    opp_lineups_eff = lineup_eff_predictions_as_array(opp_lineups_df, eff_targets)
    
    result = np.concatenate(
        [expected_sum_categories_won(lineups, opp_lineups),
         expected_efficiency_categories_won(lineups_eff, opp_lineups_eff)],
        axis=1
    )
    return result


def actual_categories_won(lineups_df, opp_lineups_df, sum_targets, eff_targets):
    lineups = lineup_stats_as_array(lineups_df, sum_targets)
    opp_lineups = lineup_stats_as_array(opp_lineups_df, sum_targets)

    lineups_eff = lineup_eff_stats_as_array(lineups_df, eff_targets)
    opp_lineups_eff = lineup_eff_stats_as_array(opp_lineups_df, eff_targets)
    
    result = np.concatenate(
        [actual_sum_categories_won(lineups, opp_lineups),
         actual_efficiency_categories_won(lineups_eff, opp_lineups_eff)],
        axis=1
    )
    return result


def test_expected_categories_won(inputs, sum_targets, eff_targets):
    lineups_df = inputs.groupby("date").apply(lambda x: x.sample(10)).reset_index(drop=True)
    opp_lineups_df = inputs.groupby("date").apply(lambda x: x.sample(10)).reset_index(drop=True)
    exp_result = expected_categories_won(
        lineups_df,
        opp_lineups_df,
        sum_targets=sum_targets,
        eff_targets=eff_targets,
    )
    act_result = actual_categories_won(
        lineups_df,
        opp_lineups_df,
        sum_targets=sum_targets,
        eff_targets=eff_targets,
    )
    cols = sum_targets + eff_targets
    result = pd.DataFrame(
        np.concatenate([exp_result, act_result], axis=1),
        columns=[f"{col}_predicted" for col in cols] + [f"{col}_actual" for col in cols]
    )
    return result
