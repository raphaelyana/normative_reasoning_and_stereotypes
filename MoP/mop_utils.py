import pandas as pd
from typing import Dict, List




def extract_top_performers_from_preliminary(result_preliminary: dict, top_n: int = 10) -> Dict[str, float]:    
    try:
        rescue_stats = result_preliminary.get('rescue_stats', pd.DataFrame())
        if isinstance(rescue_stats, pd.DataFrame) and not rescue_stats.empty:
            top_rescue = rescue_stats.nlargest(top_n, 'rescue_rate')
            
            performance_map = {}
            for _, row in top_rescue.iterrows():
                profile = row['profile']
                accuracy = row.get('profile_acc', 0.7)
                performance_map[profile] = accuracy
            
            print(f"Extracted {len(performance_map)} top performers from preliminary analysis")
            return performance_map
        else:
            print("No rescue stats found in preliminary analysis")
            return {}
        
    except Exception as e:
        print(f"Error extracting top performers: {e}")
        return {}


def extract_pareto_pool_from_tier1(result_tier1: dict) -> List[str]:
    """Extract Pareto optimal profiles from Tier 1 analysis."""
    try:
        pareto_results = result_tier1.get('pareto_results', {})
        pareto_optimal = pareto_results.get('pareto_optimal', pd.DataFrame())
        
        if isinstance(pareto_optimal, pd.DataFrame) and not pareto_optimal.empty:
            profiles = pareto_optimal['profile'].tolist()
            print(f"✓ Extracted {len(profiles)} Pareto optimal profiles from Tier 1")
            return profiles
        else:
            print("⚠ No Pareto optimal profiles found in Tier 1")
            return []
    except Exception as e:
        print(f"✗ Error extracting Pareto pool from Tier 1: {e}")
        return []

def extract_demographic_performance_map_from_tier2(result_tier2: dict) -> Dict[str, float]:
    """Extract performance by ethnicity from Tier 2 ensemble analysis."""
    try:
        ensemble_analysis = result_tier2.get('ensemble_analysis', {})
        ensemble_results = ensemble_analysis.get('ensemble_results', {})
        
        ethnicity_performance = {}
        
        for group_name, results in ensemble_results.items():
            parts = group_name.split('_')
            if len(parts) >= 2:
                ethnicity = parts[-1]
                if ethnicity == 'eastern':
                    ethnicity = 'middle_eastern'
                accuracy = results.get('accuracy', 0.7)
                ethnicity_performance[ethnicity] = accuracy
        
        if ethnicity_performance:
            print(f"  Extracted performance map for {len(ethnicity_performance)} ethnicities from Tier 2")
            return ethnicity_performance
        else:
            print("No demographic performance data found in Tier 2")
            return {}
    except Exception as e:
        print(f"Error extracting demographic performance from Tier 2: {e}")
        return {}

def extract_consistency_data_from_tier3(result_tier3: dict) -> Dict[str, Dict]:
    """Extract consistency/volatility data from Tier 3 analysis."""
    try:
        consistency_analysis = result_tier3.get('consistency_analysis', {})
        consistency_data = consistency_analysis.get('consistency_data', {})
        
        if consistency_data:
            print(f"Extracted consistency data for {len(consistency_data)} profiles from Tier 3")
            return consistency_data
        else:
            print("No consistency data found in Tier 3")
            return {}
    except Exception as e:
        print(f"✗ Error extracting consistency data from Tier 3: {e}")
        return {}

def extract_bias_recommendations_from_tier3(result_tier3: dict) -> Dict[str, float]:
    """Extract bias mitigation recommendations from Tier 3 causal analysis."""
    try:
        causal_analysis = result_tier3.get('causal_analysis', {})
        bias_detection = causal_analysis.get('bias_detection', {})
        
        recommendations = {
            'max_single_ethnicity': 0.8,  
            'max_single_gender': 0.9,     
            'max_extra_error_rate': 0.08  
        }
        
        high_bias_count = sum(1 for detection in bias_detection.values() 
                             if 'HIGH BIAS' in detection.get('bias_level', ''))
        
        if high_bias_count > 2:
            recommendations['max_single_ethnicity'] = 0.6
            recommendations['max_single_gender'] = 0.7
            print("High bias detected - using restrictive constraints")
        elif high_bias_count > 0:
            recommendations['max_single_ethnicity'] = 0.7
            recommendations['max_single_gender'] = 0.8
            print("Moderate bias detected - using balanced constraints")
        else:
            recommendations['max_single_ethnicity'] = 0.8
            recommendations['max_single_gender'] = 0.9
            print("Low bias detected - using performance-optimized constraints")
        
        return recommendations
    except Exception as e:
        print(f"✗ Error extracting bias recommendations from Tier 3: {e}")
        return {'max_single_ethnicity': 0.8, 'max_single_gender': 0.9, 'max_extra_error_rate': 0.08}