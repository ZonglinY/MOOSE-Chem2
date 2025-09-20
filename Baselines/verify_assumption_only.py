import argparse, os, sys, json
import statistics
from scipy import stats

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AssumptionVerifier(object):
    
    def __init__(self, scores_file_path):
        self.scores_file_path = scores_file_path
        self.load_scores()
    
    def load_scores(self):
        """Load the hypothesis scores from file"""
        if os.path.exists(self.scores_file_path):
            with open(self.scores_file_path, "r") as f:
                self.hyp_with_specific_concept_score_list = json.load(f)
                print(f"Loaded scores from {self.scores_file_path}")
        else:
            print(f"Error: Scores file not found at {self.scores_file_path}")
            self.hyp_with_specific_concept_score_list = {}
    
    def verify_fundamental_assumption(self):
        """
        Verify the fundamental assumption that the scores of specific concept hypotheses, 
        if averaged, should be close to the original general concept hypothesis.
        """
        if not self.hyp_with_specific_concept_score_list:
            print("No hypothesis scores available. Please check the scores file.")
            return
        
        # Results storage
        assumption_results = []
        detailed_results = {}
        
        print("="*60)
        print("VERIFYING FUNDAMENTAL ASSUMPTION")
        print("="*60)
        
        for bkg_id_key, general_concepts_dict in self.hyp_with_specific_concept_score_list.items():
            bkg_id = int(bkg_id_key)
            detailed_results[bkg_id_key] = {}
            
            # Skip if no scores for this background
            if not general_concepts_dict:
                continue
                
            print(f"\n--- Background ID: {bkg_id} ---")
            
            # For each general concept in this background
            for general_concept, specific_scores_dict in general_concepts_dict.items():
                if not specific_scores_dict:
                    continue
                    
                # Extract and convert scores to floats
                scores = []
                original_score = None
                specific_scores = {}
                
                for specific_concept, score_str in specific_scores_dict.items():
                    try:
                        score = float(score_str.strip())
                        scores.append(score)
                        specific_scores[specific_concept] = score
                        
                        # The original general concept score (where specific_concept == general_concept)
                        if specific_concept == general_concept:
                            original_score = score
                    except (ValueError, AttributeError):
                        print(f"Warning: Could not parse score '{score_str}' for {specific_concept}")
                        continue
                
                if len(scores) < 2:  # Need at least original + one specific
                    continue
                    
                # Calculate average of all specific concept scores
                avg_score = statistics.mean(scores)
                std_score = statistics.stdev(scores) if len(scores) > 1 else 0
                
                # Store detailed results
                detailed_results[bkg_id_key][general_concept] = {
                    'original_score': original_score,
                    'avg_specific_scores': avg_score,
                    'std_specific_scores': std_score,
                    'specific_scores': specific_scores,
                    'num_specific_concepts': len(scores)
                }
                
                # Calculate difference between average and original
                if original_score is not None:
                    diff_from_original = abs(avg_score - original_score)
                    
                    print(f"  General Concept: {general_concept}")
                    print(f"    Original Score: {original_score:.1f}")
                    print(f"    Average of Specific Scores: {avg_score:.1f} (±{std_score:.1f})")
                    print(f"    Difference from Original: {diff_from_original:.1f}")
                    print(f"    Number of Specific Concepts: {len(scores)}")
                    
                    # Compare with other general concepts in the same background
                    other_concept_diffs = []
                    for other_concept, other_scores_dict in general_concepts_dict.items():
                        if other_concept != general_concept and other_scores_dict:
                            # Get the original score of the other concept
                            other_original_score = None
                            for other_specific, other_score_str in other_scores_dict.items():
                                if other_specific == other_concept:
                                    try:
                                        other_original_score = float(other_score_str.strip())
                                        break
                                    except (ValueError, AttributeError):
                                        continue
                            
                            if other_original_score is not None:
                                diff_from_other = abs(avg_score - other_original_score)
                                other_concept_diffs.append(diff_from_other)
                    
                    # Store results for statistical analysis
                    result_entry = {
                        'bkg_id': bkg_id,
                        'general_concept': general_concept,
                        'original_score': original_score,
                        'avg_specific_scores': avg_score,
                        'diff_from_original': diff_from_original,
                        'diff_from_others': other_concept_diffs,
                        'closer_to_original': True,  # Will update this
                        'num_specific_concepts': len(scores)
                    }
                    
                    # Check if average is closer to original than to other concepts
                    if other_concept_diffs:
                        min_diff_from_others = min(other_concept_diffs)
                        avg_diff_from_others = statistics.mean(other_concept_diffs)
                        result_entry['min_diff_from_others'] = min_diff_from_others
                        result_entry['avg_diff_from_others'] = avg_diff_from_others
                        result_entry['closer_to_original'] = diff_from_original < min_diff_from_others
                        
                        print(f"    Min Difference from Other Concepts: {min_diff_from_others:.1f}")
                        print(f"    Avg Difference from Other Concepts: {avg_diff_from_others:.1f}")
                        print(f"    Closer to Original? {result_entry['closer_to_original']}")
                    
                    assumption_results.append(result_entry)
        
        # Statistical Analysis
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        if not assumption_results:
            print("No valid results for analysis.")
            return detailed_results
        
        # Overall statistics
        all_diffs_from_original = [r['diff_from_original'] for r in assumption_results]
        all_diffs_from_others = []
        for r in assumption_results:
            if 'diff_from_others' in r:
                all_diffs_from_others.extend(r['diff_from_others'])
        
        print(f"Total number of test cases: {len(assumption_results)}")
        print(f"Average difference from original concept: {statistics.mean(all_diffs_from_original):.2f} (±{statistics.stdev(all_diffs_from_original):.2f})")
        
        if all_diffs_from_others:
            print(f"Average difference from other concepts: {statistics.mean(all_diffs_from_others):.2f} (±{statistics.stdev(all_diffs_from_others):.2f})")
            
            # Statistical significance test
            t_stat, p_value = stats.ttest_ind(all_diffs_from_original, all_diffs_from_others)
            print(f"T-test p-value (original vs others): {p_value:.4f}")
            print(f"Statistically significant difference? {p_value < 0.05}")
        
        # Fraction of cases where assumption holds
        cases_closer_to_original = [r for r in assumption_results if r.get('closer_to_original', False)]
        fraction_supporting = len(cases_closer_to_original) / len(assumption_results)
        print(f"Fraction of cases where avg is closer to original: {fraction_supporting:.2f} ({len(cases_closer_to_original)}/{len(assumption_results)})")
        
        # Distribution analysis
        print(f"\nDifference from original - Distribution:")
        print(f"  Min: {min(all_diffs_from_original):.1f}")
        print(f"  Q25: {statistics.quantiles(all_diffs_from_original, n=4)[0]:.1f}")
        print(f"  Median: {statistics.median(all_diffs_from_original):.1f}")
        print(f"  Q75: {statistics.quantiles(all_diffs_from_original, n=4)[2]:.1f}")
        print(f"  Max: {max(all_diffs_from_original):.1f}")
        
        # Save detailed results
        results_save_path = "./Baselines/fundamental_assumption_verification_results.json"
        with open(results_save_path, "w") as f:
            json.dump({
                'detailed_results': detailed_results,
                'statistical_summary': {
                    'total_cases': len(assumption_results),
                    'avg_diff_from_original': statistics.mean(all_diffs_from_original),
                    'std_diff_from_original': statistics.stdev(all_diffs_from_original),
                    'avg_diff_from_others': statistics.mean(all_diffs_from_others) if all_diffs_from_others else None,
                    'std_diff_from_others': statistics.stdev(all_diffs_from_others) if len(all_diffs_from_others) > 1 else None,
                    'fraction_supporting_assumption': fraction_supporting,
                    'cases_supporting_assumption': len(cases_closer_to_original),
                    't_test_p_value': p_value if all_diffs_from_others else None
                }
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_save_path}")
        
        # Conclusion
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        
        if fraction_supporting > 0.7:
            print("✓ ASSUMPTION STRONGLY SUPPORTED: The majority of cases show that averaged")
            print("  specific concept scores are closer to the original general concept than")
            print("  to other general concepts.")
        elif fraction_supporting > 0.5:
            print("~ ASSUMPTION MODERATELY SUPPORTED: More than half of the cases support")
            print("  the assumption, but the evidence is not overwhelming.")
        else:
            print("✗ ASSUMPTION NOT SUPPORTED: The majority of cases do not support the")
            print("  fundamental assumption.")
        
        if all_diffs_from_others and statistics.mean(all_diffs_from_original) < statistics.mean(all_diffs_from_others):
            print("✓ STATISTICAL EVIDENCE: On average, specific concept scores are indeed")
            print("  closer to their original general concept than to other concepts.")
        
        return detailed_results

    def verify_relaxed_assumption(self, closeness_thresholds=[3, 5, 7, 10]):
        """
        Verify a relaxed assumption: specific concept scores should be close to 
        the original general concept, regardless of comparison to other concepts.
        """
        if not self.hyp_with_specific_concept_score_list:
            print("No hypothesis scores available. Please check the scores file.")
            return
        
        print("\n" + "="*60)
        print("VERIFYING RELAXED ASSUMPTION (CLOSENESS ONLY)")
        print("="*60)
        
        all_differences = []
        detailed_cases = []
        
        for bkg_id_key, general_concepts_dict in self.hyp_with_specific_concept_score_list.items():
            if not general_concepts_dict:
                continue
                
            for general_concept, specific_scores_dict in general_concepts_dict.items():
                if not specific_scores_dict:
                    continue
                    
                # Extract and convert scores to floats
                scores = []
                original_score = None
                
                for specific_concept, score_str in specific_scores_dict.items():
                    try:
                        score = float(score_str.strip())
                        scores.append(score)
                        
                        if specific_concept == general_concept:
                            original_score = score
                    except (ValueError, AttributeError):
                        continue
                
                if len(scores) < 2 or original_score is None:
                    continue
                
                # Calculate average and difference
                avg_score = statistics.mean(scores)
                diff_from_original = abs(avg_score - original_score)
                all_differences.append(diff_from_original)
                
                # ALSO calculate excluding the original concept
                specific_only_scores = []
                for concept, score_str in specific_scores_dict.items():
                    if concept != general_concept:
                        try:
                            specific_only_scores.append(float(score_str.strip()))
                        except (ValueError, AttributeError):
                            continue
                if len(specific_only_scores) > 0:
                    avg_specific_only = statistics.mean(specific_only_scores)
                    diff_specific_only = abs(avg_specific_only - original_score)
                else:
                    avg_specific_only = None
                    diff_specific_only = None
                
                # Check against different thresholds
                case_info = {
                    'bkg_id': int(bkg_id_key),
                    'general_concept': general_concept,
                    'original_score': original_score,
                    'avg_specific_scores': avg_score,  # includes original
                    'avg_specific_only': avg_specific_only,  # excludes original
                    'difference': diff_from_original,  # includes original in avg
                    'difference_specific_only': diff_specific_only,  # excludes original
                    'num_specific_concepts': len(scores)
                }
                detailed_cases.append(case_info)
        
        print(f"Total test cases analyzed: {len(all_differences)}")
        print(f"Average difference from original: {statistics.mean(all_differences):.2f} ± {statistics.stdev(all_differences):.2f}")
        print(f"Median difference: {statistics.median(all_differences):.2f}")
        
        # ALSO analyze excluding original from specific concept average
        specific_only_differences = [case['difference_specific_only'] for case in detailed_cases 
                                   if case['difference_specific_only'] is not None]
        
        if specific_only_differences:
            print(f"\n--- COMPARISON: Including vs Excluding Original in Average ---")
            print(f"Including original in average:")
            print(f"  Average difference: {statistics.mean(all_differences):.2f} ± {statistics.stdev(all_differences):.2f}")
            print(f"  Median difference:  {statistics.median(all_differences):.2f}")
            
            print(f"Excluding original from average:")
            print(f"  Average difference: {statistics.mean(specific_only_differences):.2f} ± {statistics.stdev(specific_only_differences):.2f}")
            print(f"  Median difference:  {statistics.median(specific_only_differences):.2f}")
            
            print(f"Effect of including original: {statistics.mean(specific_only_differences) - statistics.mean(all_differences):.2f} points smaller differences")
        
        # Use specific-only differences for the main analysis
        analysis_differences = specific_only_differences if specific_only_differences else all_differences
        valid_cases = [case for case in detailed_cases if case['difference_specific_only'] is not None] if specific_only_differences else detailed_cases
        
        print(f"\n" + "="*60)
        print("MAIN ANALYSIS (EXCLUDING ORIGINAL FROM SPECIFIC AVERAGE)")
        print("="*60)
        print(f"Total test cases analyzed: {len(analysis_differences)}")
        print(f"Average difference from original: {statistics.mean(analysis_differences):.2f} ± {statistics.stdev(analysis_differences):.2f}")
        print(f"Median difference: {statistics.median(analysis_differences):.2f}")
        
        # Distribution analysis
        print(f"\nDifference Distribution:")
        print(f"  Min: {min(analysis_differences):.1f}")
        print(f"  Q25: {statistics.quantiles(analysis_differences, n=4)[0]:.1f}")
        print(f"  Median: {statistics.median(analysis_differences):.1f}")
        print(f"  Q75: {statistics.quantiles(analysis_differences, n=4)[2]:.1f}")
        print(f"  Max: {max(analysis_differences):.1f}")
        
        # Recalculate thresholds using specific-only differences
        results_by_threshold = {threshold: [] for threshold in closeness_thresholds}
        for case in valid_cases:
            diff = case['difference_specific_only']
            for threshold in closeness_thresholds:
                is_close = diff <= threshold
                results_by_threshold[threshold].append(is_close)
        
        # Analyze by thresholds
        print(f"\n" + "="*50)
        print("CLOSENESS ANALYSIS BY THRESHOLD")
        print("="*50)
        
        threshold_results = {}
        for threshold in closeness_thresholds:
            close_cases = sum(results_by_threshold[threshold])
            total_cases = len(results_by_threshold[threshold])
            fraction = close_cases / total_cases if total_cases > 0 else 0
            threshold_results[threshold] = {
                'close_cases': close_cases,
                'total_cases': total_cases,
                'fraction': fraction
            }
            
            print(f"Within {threshold:2d} points: {close_cases:3d}/{total_cases} ({fraction:.1%})")
        
        # Find optimal threshold
        print(f"\n" + "="*50)
        print("DETAILED ANALYSIS")
        print("="*50)
        
        # What percentage are within different ranges?
        ranges = [1, 2, 3, 4, 5, 7, 10, 15, 20]
        print("Cumulative closeness distribution:")
        for r in ranges:
            within_range = sum(1 for d in analysis_differences if d <= r)
            percentage = within_range / len(analysis_differences) * 100
            print(f"  ≤ {r:2d} points: {percentage:5.1f}% ({within_range}/{len(analysis_differences)})")
        
        # Random baseline comparison using specific-only data
        print(f"\n" + "="*50)
        print("BASELINE COMPARISON")
        print("="*50)
        
        # Create random baseline: what if we randomly paired averages with original scores?
        import random
        random.seed(42)  # For reproducibility
        
        all_averages = [case['avg_specific_only'] for case in valid_cases if case['avg_specific_only'] is not None]
        all_originals = [case['original_score'] for case in valid_cases if case['avg_specific_only'] is not None]
        
        # Shuffle averages to create random pairing
        random_averages = all_averages.copy()
        random.shuffle(random_averages)
        
        random_differences = [abs(avg - orig) for avg, orig in zip(random_averages, all_originals)]
        
        print(f"Actual average difference:    {statistics.mean(analysis_differences):.2f}")
        print(f"Random baseline difference:   {statistics.mean(random_differences):.2f}")
        print(f"Improvement over random:      {statistics.mean(random_differences) - statistics.mean(analysis_differences):.2f} points")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(analysis_differences, random_differences)
        print(f"T-test vs random (p-value):   {p_value:.6f}")
        print(f"Significantly better than random: {p_value < 0.05}")
        
        # Best threshold recommendation
        print(f"\n" + "="*50)
        print("RECOMMENDATIONS")
        print("="*50)
        
        # Find threshold where we get reasonable coverage (e.g., >70%)
        recommended_threshold = None
        for threshold in sorted(closeness_thresholds):
            if threshold_results[threshold]['fraction'] >= 0.7:
                recommended_threshold = threshold
                break
        
        if recommended_threshold:
            print(f"✓ RECOMMENDED THRESHOLD: {recommended_threshold} points")
            print(f"  This captures {threshold_results[recommended_threshold]['fraction']:.1%} of cases")
            print(f"  Interpretation: Specific concept hypothesis scores (excluding original)")
            print(f"  are typically within {recommended_threshold} points of their general concept")
        else:
            # Find the best threshold that gives maximum coverage
            best_threshold = max(closeness_thresholds, key=lambda t: threshold_results[t]['fraction'])
            print(f"~ BEST AVAILABLE: {best_threshold} points")
            print(f"  This captures {threshold_results[best_threshold]['fraction']:.1%} of cases")
            print(f"  Note: No threshold achieves 70% coverage")
        
        # Save results with specific-only focus
        relaxed_results = {
            'analysis_type': 'excluding_original_from_specific_average',
            'all_differences': analysis_differences,
            'threshold_analysis': threshold_results,
            'detailed_cases': valid_cases,
            'baseline_comparison': {
                'actual_avg_diff': statistics.mean(analysis_differences),
                'random_avg_diff': statistics.mean(random_differences),
                'improvement': statistics.mean(random_differences) - statistics.mean(analysis_differences),
                't_test_p_value': p_value
            },
            'recommended_threshold': recommended_threshold,
            'comparison_with_including_original': {
                'including_avg_diff': statistics.mean(all_differences),
                'excluding_avg_diff': statistics.mean(analysis_differences),
                'bias_effect': statistics.mean(analysis_differences) - statistics.mean(all_differences)
            }
        }
        
        results_save_path = "./Baselines/relaxed_assumption_verification_results.json"
        with open(results_save_path, "w") as f:
            json.dump(relaxed_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_save_path}")
        
        return relaxed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify fundamental assumption using existing scores')
    parser.add_argument("--scores_file", type=str, default="./Baselines/hyp_with_specific_concept_score_list.json", 
                       help="Path to the scores file")
    parser.add_argument("--relaxed_only", action="store_true", help="Only run the relaxed assumption analysis")
    args = parser.parse_args()

    verifier = AssumptionVerifier(args.scores_file)
    
    if args.relaxed_only:
        verifier.verify_relaxed_assumption()
    else:
        verifier.verify_fundamental_assumption()
        verifier.verify_relaxed_assumption() 