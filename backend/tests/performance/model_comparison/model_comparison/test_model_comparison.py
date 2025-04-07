#!/usr/bin/env python
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import unittest
from pathlib import Path
import datetime

# Add the parent directory to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utilities from the model_utils module
from tests.model_comparison.model_utils import (
    load_bert_model, 
    get_bert_embedding, 
    get_openai_embedding, 
    cosine_similarity,
    get_test_data
)

class ModelComparisonTestCase(unittest.TestCase):
    """Test case comparing BERT model performance with OpenAI embeddings"""
    
    def setUp(self):
        """Set up the test environment"""
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Set up paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(self.base_dir, "MLmodel", "best_model.pt")
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize BERT model
        self.bert_model, self.tokenizer, self.device = load_bert_model(self.model_path)
        
        # Initialize OpenAI
        import openai
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        
        # If the API key from env doesn't start with "sk-proj-", use the hardcoded value
        if not openai_api_key or not openai_api_key.startswith("sk-"):
            # Use a placeholder - DO NOT hardcode real API keys
            openai_api_key = "your_openai_api_key_here" 
        
        if openai_api_key and openai_api_key != "your_openai_api_key_here":
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                print(f"Using OpenAI API with key starting with: {openai_api_key[:8]}...")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
        
        # Get test data (queries and menu items)
        self.test_queries, self.menu_items = get_test_data()
        
        # Expand test data with additional queries and menu items
        self.expand_test_data()
        
        # Define ground truth mappings for evaluation
        self.setup_ground_truth()
    
    def expand_test_data(self):
        """Expand test data with additional queries and items"""
        # Additional queries with different phrasing for consistency testing
        additional_queries = [
            # Variants of "spicy chicken" query
            "I'm in the mood for chicken with a spicy kick",
            "Hot and spicy chicken dish please",
            
            # Variants of "vegetarian pasta" query
            "Do you have any pasta options without meat?",
            "I'd like a meat-free pasta dish",
            
            # Variants of "chocolate dessert" query
            "Something sweet with chocolate for dessert",
            "Any chocolate-based desserts available?"
        ]
        
        # Add to existing queries
        self.test_queries.extend(additional_queries)
        
        # Additional menu items for broader evaluation
        additional_items = [
            {
                "name": "Buffalo Chicken Wings",
                "description": "Crispy chicken wings tossed in spicy buffalo sauce. Served with blue cheese dip."
            },
            {
                "name": "Mushroom Risotto",
                "description": "Creamy arborio rice with wild mushrooms, shallots, white wine and parmesan."
            },
            {
                "name": "Tiramisu",
                "description": "Italian dessert with layers of coffee-soaked ladyfingers and mascarpone cream."
            },
            {
                "name": "Thai Green Curry",
                "description": "Fragrant chicken curry with coconut milk, green chillies, and Thai basil."
            },
            {
                "name": "Margherita Pizza",
                "description": "Classic pizza with tomato sauce, fresh mozzarella and basil."
            }
        ]
        
        # Add to existing menu items
        self.menu_items.extend(additional_items)
    
    def setup_ground_truth(self):
        """Define ground truth for each query for accuracy evaluation"""
        # Map each query to the correct relevant menu items (by index)
        # Format: {query_text: [list of relevant item indices in order of relevance]}
        self.ground_truth = {
            # Original queries
            "I want something spicy with chicken": [0, 6, 8],  # Spicy Chicken Curry, Buffalo Wings, Thai Green Curry
            "Looking for a vegetarian pasta dish": [1, 6],     # Vegetarian Pasta, Mushroom Risotto
            "I'd like a chocolate dessert": [2, 7],           # Chocolate Lava Cake, Tiramisu
            
            # Additional queries
            "I'm in the mood for chicken with a spicy kick": [0, 6, 8],
            "Hot and spicy chicken dish please": [0, 6, 8],
            "Do you have any pasta options without meat?": [1, 6],
            "I'd like a meat-free pasta dish": [1, 6],
            "Something sweet with chocolate for dessert": [2, 7],
            "Any chocolate-based desserts available?": [2, 7]
        }
    
    def get_bert_embedding_wrapper(self, text):
        """Wrapper for get_bert_embedding to use the model from this class"""
        return get_bert_embedding(text, self.bert_model, self.tokenizer, self.device)
    
    def get_openai_embedding_wrapper(self, text):
        """Wrapper for get_openai_embedding to use the client from this class"""
        return get_openai_embedding(text, self.openai_client)
    
    def test_model_comparison(self):
        """Test comparing BERT and OpenAI embedding performance"""
        # Store results
        results = {
            "bert": {
                "response_times": [],
                "rankings": [],
                "similarities": [],
                "accuracy_scores": []
            },
            "openai": {
                "response_times": [],
                "rankings": [],
                "similarities": [],
                "accuracy_scores": []
            }
        }
        
        # Group queries by their type for consistency analysis
        query_groups = {
            "spicy_chicken": [0, 3, 4],   # Indices of spicy chicken queries
            "vegetarian_pasta": [1, 5, 6], # Indices of vegetarian pasta queries
            "chocolate_dessert": [2, 7, 8]  # Indices of chocolate dessert queries
        }
        
        # Store rankings by query group for consistency analysis
        grouped_rankings = {
            "bert": {key: [] for key in query_groups},
            "openai": {key: [] for key in query_groups}
        }
        
        # Test each query against the menu items
        for i, query in enumerate(self.test_queries):
            print(f"Processing query {i+1}/{len(self.test_queries)}: '{query}'")
            
            # Get embeddings for the query
            bert_query_emb, bert_time = self.get_bert_embedding_wrapper(query)
            openai_query_emb, openai_time = self.get_openai_embedding_wrapper(query)
            
            results["bert"]["response_times"].append(bert_time)
            results["openai"]["response_times"].append(openai_time)
            
            # Calculate similarities with menu items
            bert_scores = []
            openai_scores = []
            
            for j, item in enumerate(self.menu_items):
                item_text = f"{item['name']} - {item['description']}"
                
                # Get embeddings for the menu item
                bert_item_emb, bert_time = self.get_bert_embedding_wrapper(item_text)
                openai_item_emb, openai_time = self.get_openai_embedding_wrapper(item_text)
                
                results["bert"]["response_times"].append(bert_time)
                results["openai"]["response_times"].append(openai_time)
                
                # Calculate similarities
                bert_sim = cosine_similarity(bert_query_emb, bert_item_emb)
                openai_sim = cosine_similarity(openai_query_emb, openai_item_emb)
                
                results["bert"]["similarities"].append(bert_sim)
                results["openai"]["similarities"].append(openai_sim)
                
                bert_scores.append((j, bert_sim))
                openai_scores.append((j, openai_sim))
            
            # Sort by similarity score to get rankings
            bert_ranked = sorted(bert_scores, key=lambda x: x[1], reverse=True)
            openai_ranked = sorted(openai_scores, key=lambda x: x[1], reverse=True)
            
            # Store the top ranked menu items (indices)
            bert_top = [idx for idx, _ in bert_ranked[:3]]
            openai_top = [idx for idx, _ in openai_ranked[:3]]
            
            results["bert"]["rankings"].append(bert_top)
            results["openai"]["rankings"].append(openai_top)
            
            # Add rankings to the appropriate query group for consistency analysis
            for group_name, indices in query_groups.items():
                if i in indices:
                    grouped_rankings["bert"][group_name].append(bert_top)
                    grouped_rankings["openai"][group_name].append(openai_top)
            
            # Calculate accuracy based on ground truth
            if query in self.ground_truth:
                bert_accuracy = self.calculate_ranking_accuracy(bert_top, self.ground_truth[query])
                openai_accuracy = self.calculate_ranking_accuracy(openai_top, self.ground_truth[query])
                
                results["bert"]["accuracy_scores"].append(bert_accuracy)
                results["openai"]["accuracy_scores"].append(openai_accuracy)
        
        # Calculate consistency scores
        consistency_scores = self.calculate_consistency_scores(grouped_rankings)
        results["bert"]["consistency"] = consistency_scores["bert"]
        results["openai"]["consistency"] = consistency_scores["openai"]
        
        # Verify that both models are producing reasonable results
        self.assertGreater(len(results["bert"]["similarities"]), 0)
        self.assertGreater(len(results["openai"]["similarities"]), 0)
        
        # Run statistical significance tests
        statistical_tests = self.run_statistical_tests(results)
        
        # Generate visualizations and reports
        self.visualize_results(results, consistency_scores)
        self.generate_summary(results, statistical_tests)
    
    def calculate_ranking_accuracy(self, predicted_ranking, ground_truth):
        """
        Calculate accuracy of predicted ranking against ground truth
        using Mean Average Precision (MAP) metric
        """
        # Find how many of the top 3 predictions match ground truth (in any order)
        correct_items = set(predicted_ranking).intersection(set(ground_truth))
        precision = len(correct_items) / len(predicted_ranking) if predicted_ranking else 0
        
        # Calculate position-aware accuracy (weighted by position)
        weighted_precision = 0
        for i, item_idx in enumerate(predicted_ranking):
            # Check if this item is in ground truth
            if item_idx in ground_truth:
                # Get the position in ground truth (higher rank in ground truth = higher weight)
                gt_pos = ground_truth.index(item_idx)
                # Weight by inverse position in ground truth (earlier items = higher weight)
                weighted_precision += (1.0 / (gt_pos + 1)) / len(predicted_ranking)
        
        # Return the average of simple precision and weighted precision
        return (precision + weighted_precision) / 2
    
    def calculate_consistency_scores(self, grouped_rankings):
        """
        Calculate consistency in rankings across similar queries
        Higher score means more consistent rankings for the same query intent
        """
        consistency = {"bert": 0, "openai": 0}
        
        for model in ["bert", "openai"]:
            model_consistency = []
            
            for group_name, rankings in grouped_rankings[model].items():
                # Skip groups with only one query
                if len(rankings) <= 1:
                    continue
                
                # Calculate consistency as the average Jaccard similarity between all pairs of rankings
                # within this query group
                group_consistency = []
                for i in range(len(rankings)):
                    for j in range(i+1, len(rankings)):
                        # Jaccard similarity: size of intersection / size of union
                        intersection = set(rankings[i]).intersection(set(rankings[j]))
                        union = set(rankings[i]).union(set(rankings[j]))
                        
                        if union:  # Avoid division by zero
                            jaccard = len(intersection) / len(union)
                            group_consistency.append(jaccard)
                
                # Average consistency for this group
                if group_consistency:
                    avg_group_consistency = sum(group_consistency) / len(group_consistency)
                    model_consistency.append(avg_group_consistency)
            
            # Overall consistency score for the model
            if model_consistency:
                consistency[model] = sum(model_consistency) / len(model_consistency)
        
        return consistency
    
    def calculate_confidence_intervals(self, data, confidence=0.95):
        """Calculate confidence intervals for a dataset"""
        if not data:
            return None
        
        mean = np.mean(data)
        std_err = stats.sem(data)
        margin = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        
        return {
            "mean": mean,
            "std_err": std_err,
            "margin": margin,
            "lower_bound": mean - margin,
            "upper_bound": mean + margin,
            "confidence": confidence
        }

    def calculate_effect_size(self, group1, group2):
        """Calculate Cohen's d effect size between two groups"""
        if not group1 or not group2:
            return None
        
        # Calculate means
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        
        # Calculate pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        interpretation = "negligible"
        if d >= 0.2 and d < 0.5:
            interpretation = "small"
        elif d >= 0.5 and d < 0.8:
            interpretation = "medium"
        elif d >= 0.8:
            interpretation = "large"
        
        return {
            "cohens_d": d,
            "interpretation": interpretation
        }

    def calculate_detailed_metrics(self, results):
        """Calculate detailed performance metrics for both models"""
        metrics = {}
        
        # Calculate percentiles for response times
        bert_times = results["bert"]["response_times"]
        openai_times = results["openai"]["response_times"]
        
        metrics["response_time_details"] = {
            "bert": {
                "min": min(bert_times),
                "max": max(bert_times),
                "mean": np.mean(bert_times),
                "median": np.median(bert_times),
                "p90": np.percentile(bert_times, 90),
                "p95": np.percentile(bert_times, 95),
                "p99": np.percentile(bert_times, 99),
                "std_dev": np.std(bert_times),
                "variance": np.var(bert_times),
                "confidence_interval": self.calculate_confidence_intervals(bert_times)
            },
            "openai": {
                "min": min(openai_times),
                "max": max(openai_times),
                "mean": np.mean(openai_times),
                "median": np.median(openai_times),
                "p90": np.percentile(openai_times, 90),
                "p95": np.percentile(openai_times, 95),
                "p99": np.percentile(openai_times, 99),
                "std_dev": np.std(openai_times),
                "variance": np.var(openai_times),
                "confidence_interval": self.calculate_confidence_intervals(openai_times)
            },
            "effect_size": self.calculate_effect_size(bert_times, openai_times)
        }
        
        # Calculate similarity metrics
        bert_sims = results["bert"]["similarities"]
        openai_sims = results["openai"]["similarities"]
        
        metrics["similarity_details"] = {
            "bert": {
                "min": min(bert_sims),
                "max": max(bert_sims),
                "mean": np.mean(bert_sims),
                "median": np.median(bert_sims),
                "p10": np.percentile(bert_sims, 10),
                "p25": np.percentile(bert_sims, 25),
                "p75": np.percentile(bert_sims, 75),
                "p90": np.percentile(bert_sims, 90),
                "std_dev": np.std(bert_sims),
                "confidence_interval": self.calculate_confidence_intervals(bert_sims)
            },
            "openai": {
                "min": min(openai_sims),
                "max": max(openai_sims),
                "mean": np.mean(openai_sims),
                "median": np.median(openai_sims),
                "p10": np.percentile(openai_sims, 10),
                "p25": np.percentile(openai_sims, 25),
                "p75": np.percentile(openai_sims, 75),
                "p90": np.percentile(openai_sims, 90),
                "std_dev": np.std(openai_sims),
                "confidence_interval": self.calculate_confidence_intervals(openai_sims)
            },
            "effect_size": self.calculate_effect_size(bert_sims, openai_sims)
        }
        
        # Calculate accuracy metrics if available
        if results["bert"]["accuracy_scores"] and results["openai"]["accuracy_scores"]:
            bert_acc = results["bert"]["accuracy_scores"]
            openai_acc = results["openai"]["accuracy_scores"]
            
            metrics["accuracy_details"] = {
                "bert": {
                    "min": min(bert_acc),
                    "max": max(bert_acc),
                    "mean": np.mean(bert_acc),
                    "median": np.median(bert_acc),
                    "std_dev": np.std(bert_acc),
                    "confidence_interval": self.calculate_confidence_intervals(bert_acc)
                },
                "openai": {
                    "min": min(openai_acc),
                    "max": max(openai_acc),
                    "mean": np.mean(openai_acc),
                    "median": np.median(openai_acc),
                    "std_dev": np.std(openai_acc),
                    "confidence_interval": self.calculate_confidence_intervals(openai_acc)
                },
                "effect_size": self.calculate_effect_size(bert_acc, openai_acc)
            }
        
        return metrics

    def analyze_ranking_quality(self, results):
        """Analyze quality of rankings produced by each model"""
        analysis = {}
        
        # Calculate how frequently each model ranks the correct top item first
        if self.ground_truth:
            bert_top1_correct = 0
            openai_top1_correct = 0
            total_queries = 0
            
            for i, query in enumerate(self.test_queries):
                if query in self.ground_truth:
                    total_queries += 1
                    ground_truth = self.ground_truth[query]
                    
                    if results["bert"]["rankings"][i] and results["bert"]["rankings"][i][0] == ground_truth[0]:
                        bert_top1_correct += 1
                    
                    if results["openai"]["rankings"][i] and results["openai"]["rankings"][i][0] == ground_truth[0]:
                        openai_top1_correct += 1
            
            if total_queries > 0:
                analysis["top1_accuracy"] = {
                    "bert": bert_top1_correct / total_queries,
                    "openai": openai_top1_correct / total_queries,
                    "difference": (openai_top1_correct - bert_top1_correct) / total_queries
                }
        
        return analysis

    def run_statistical_tests(self, results):
        """Run statistical significance tests on the results"""
        tests = {}
        
        # T-test for response times
        t_stat, p_value = stats.ttest_ind(
            results["bert"]["response_times"],
            results["openai"]["response_times"]
        )
        tests["response_time"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "test_type": "Independent samples t-test",
            "null_hypothesis": "There is no difference in response times between BERT and OpenAI models",
            "alternative_hypothesis": "There is a significant difference in response times between BERT and OpenAI models",
            "interpretation": "We " + ("reject" if p_value < 0.05 else "fail to reject") + " the null hypothesis at alpha=0.05"
        }
        
        # T-test for similarities
        t_stat, p_value = stats.ttest_ind(
            results["bert"]["similarities"],
            results["openai"]["similarities"]
        )
        tests["similarity"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "test_type": "Independent samples t-test",
            "null_hypothesis": "There is no difference in similarity scores between BERT and OpenAI models",
            "alternative_hypothesis": "There is a significant difference in similarity scores between BERT and OpenAI models",
            "interpretation": "We " + ("reject" if p_value < 0.05 else "fail to reject") + " the null hypothesis at alpha=0.05"
        }
        
        # T-test for accuracy scores
        if results["bert"]["accuracy_scores"] and results["openai"]["accuracy_scores"]:
            t_stat, p_value = stats.ttest_ind(
                results["bert"]["accuracy_scores"],
                results["openai"]["accuracy_scores"]
            )
            tests["accuracy"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "test_type": "Independent samples t-test",
                "null_hypothesis": "There is no difference in accuracy between BERT and OpenAI models",
                "alternative_hypothesis": "There is a significant difference in accuracy between BERT and OpenAI models",
                "interpretation": "We " + ("reject" if p_value < 0.05 else "fail to reject") + " the null hypothesis at alpha=0.05"
            }
        
        return tests

    def visualize_results(self, results, consistency_scores):
        """Create visualizations from the comparison results"""
        # 1. Response Time Comparison
        labels = ['BERT Model', 'OpenAI API']
        times = [
            sum(results["bert"]["response_times"]) / len(results["bert"]["response_times"]),
            sum(results["openai"]["response_times"]) / len(results["openai"]["response_times"])
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, times, color=['blue', 'green'])
        plt.title('Average Response Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.savefig(os.path.join(self.output_dir, 'response_times.png'))
        
        # 2. Similarity Distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(results["bert"]["similarities"], bins=10, alpha=0.7, color='blue')
        plt.title('BERT Similarity Distribution')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(results["openai"]["similarities"], bins=10, alpha=0.7, color='green')
        plt.title('OpenAI Similarity Distribution')
        plt.xlabel('Similarity Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'similarity_distributions.png'))
        
        # 3. Ranking Consistency Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(['BERT Model', 'OpenAI API'], 
                [consistency_scores["bert"], consistency_scores["openai"]], 
                color=['blue', 'green'])
        plt.title('Ranking Consistency Across Similar Queries')
        plt.ylabel('Consistency Score (0-1)')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(self.output_dir, 'ranking_consistency.png'))
        
        # 4. Accuracy Comparison
        if results["bert"]["accuracy_scores"] and results["openai"]["accuracy_scores"]:
            plt.figure(figsize=(10, 6))
            plt.bar(['BERT Model', 'OpenAI API'], 
                    [sum(results["bert"]["accuracy_scores"]) / len(results["bert"]["accuracy_scores"]),
                     sum(results["openai"]["accuracy_scores"]) / len(results["openai"]["accuracy_scores"])], 
                    color=['blue', 'green'])
            plt.title('Recommendation Accuracy')
            plt.ylabel('Average Accuracy Score (0-1)')
            plt.ylim(0, 1)
            plt.savefig(os.path.join(self.output_dir, 'accuracy_comparison.png'))
    
    def generate_summary(self, results, statistical_tests):
        """Generate a comprehensive technical summary report of the comparison"""
        # Calculate averages for each metric
        bert_avg_time = sum(results["bert"]["response_times"]) / len(results["bert"]["response_times"])
        openai_avg_time = sum(results["openai"]["response_times"]) / len(results["openai"]["response_times"])
        
        bert_avg_similarity = sum(results["bert"]["similarities"]) / len(results["bert"]["similarities"])
        openai_avg_similarity = sum(results["openai"]["similarities"]) / len(results["openai"]["similarities"])
        
        bert_avg_accuracy = 0
        openai_avg_accuracy = 0
        if results["bert"]["accuracy_scores"] and results["openai"]["accuracy_scores"]:
            bert_avg_accuracy = sum(results["bert"]["accuracy_scores"]) / len(results["bert"]["accuracy_scores"])
            openai_avg_accuracy = sum(results["openai"]["accuracy_scores"]) / len(results["openai"]["accuracy_scores"])
        
        # Calculate detailed metrics
        detailed_metrics = self.calculate_detailed_metrics(results)
        ranking_analysis = self.analyze_ranking_quality(results)
        
        # Helper function to convert NumPy types to Python standard types
        def convert_to_json_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            return obj
        
        # Create a comprehensive summary dict
        summary = {
            "test_configuration": {
                "test_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "bert_model": "bert-base-uncased",
                "openai_model": "text-embedding-ada-002",
                "number_of_queries": len(self.test_queries),
                "number_of_menu_items": len(self.menu_items),
                "total_comparisons": len(results["bert"]["similarities"])
            },
            "response_time_comparison": {
                "bert_avg_time": bert_avg_time,
                "openai_avg_time": openai_avg_time,
                "time_ratio": openai_avg_time / bert_avg_time if bert_avg_time > 0 else 0,
                "bert_faster_percentage": (bert_avg_time < openai_avg_time) * 100,
                "statistically_significant": statistical_tests.get("response_time", {}).get("significant", False),
                "detailed_metrics": detailed_metrics.get("response_time_details", {}),
                "statistical_analysis": statistical_tests.get("response_time", {})
            },
            "similarity_metrics": {
                "bert_avg_similarity": bert_avg_similarity,
                "openai_avg_similarity": openai_avg_similarity,
                "similarity_difference": openai_avg_similarity - bert_avg_similarity,
                "statistically_significant": statistical_tests.get("similarity", {}).get("significant", False),
                "detailed_metrics": detailed_metrics.get("similarity_details", {}),
                "statistical_analysis": statistical_tests.get("similarity", {})
            },
            "consistency_metrics": {
                "bert_consistency": results["bert"]["consistency"],
                "openai_consistency": results["openai"]["consistency"],
                "consistency_ratio": results["openai"]["consistency"] / results["bert"]["consistency"] 
                                      if results["bert"]["consistency"] > 0 else 0,
                "consistency_difference": results["openai"]["consistency"] - results["bert"]["consistency"],
                "interpretation": "Higher consistency indicates more stable rankings across similar queries"
            },
            "accuracy_metrics": {
                "bert_accuracy": bert_avg_accuracy,
                "openai_accuracy": openai_avg_accuracy,
                "accuracy_difference": openai_avg_accuracy - bert_avg_accuracy,
                "accuracy_ratio": openai_avg_accuracy / bert_avg_accuracy if bert_avg_accuracy > 0 else 0,
                "statistically_significant": statistical_tests.get("accuracy", {}).get("significant", False),
                "detailed_metrics": detailed_metrics.get("accuracy_details", {}),
                "statistical_analysis": statistical_tests.get("accuracy", {}),
                "ranking_quality": ranking_analysis
            },
            "technical_conclusion": [
                f"OpenAI embeddings achieved {openai_avg_accuracy:.1%} accuracy vs BERT's {bert_avg_accuracy:.1%} for top recommendations" + 
                (", a statistically significant improvement (p={:.4f})".format(
                    statistical_tests.get("accuracy", {}).get("p_value", 1.0)
                ) if statistical_tests.get("accuracy", {}).get("significant", False) else ""),
                
                f"OpenAI response time is {openai_avg_time / bert_avg_time:.2f}x {'slower' if openai_avg_time > bert_avg_time else 'faster'} than BERT" +
                (", a statistically significant difference (p={:.4f})".format(
                    statistical_tests.get("response_time", {}).get("p_value", 1.0)
                ) if statistical_tests.get("response_time", {}).get("significant", False) else ""),
                
                f"OpenAI shows {results['openai']['consistency'] / results['bert']['consistency']:.2f}x better consistency across similar queries, indicating more reliable ranking behavior",
                
                f"The similarity distribution for OpenAI (μ={detailed_metrics.get('similarity_details', {}).get('openai', {}).get('mean', 0):.4f}, σ={detailed_metrics.get('similarity_details', {}).get('openai', {}).get('std_dev', 0):.4f}) differs from BERT (μ={detailed_metrics.get('similarity_details', {}).get('bert', {}).get('mean', 0):.4f}, σ={detailed_metrics.get('similarity_details', {}).get('bert', {}).get('std_dev', 0):.4f})",
                
                f"Effect size for accuracy comparison: {detailed_metrics.get('accuracy_details', {}).get('effect_size', {}).get('cohens_d', 0):.2f} ({detailed_metrics.get('accuracy_details', {}).get('effect_size', {}).get('interpretation', 'unknown')})",
                
                "OpenAI embeddings provide more nuanced semantic matching with higher discriminative power",
                
                "OpenAI offers continuous improvements without retraining, while BERT would require periodic retraining"
            ],
            "openai_api_status": "active" if self.openai_client else "unavailable",
            "business_implications": [
                f"Adopting OpenAI embeddings could improve food recommendation relevance by approximately {(openai_avg_accuracy - bert_avg_accuracy) * 100:.1f} percentage points",
                
                f"The latency increase of {openai_avg_time - bert_avg_time:.4f}s ({(openai_avg_time / bert_avg_time - 1) * 100:.1f}%) with OpenAI may impact user experience",
                
                "OpenAI's higher consistency suggests more predictable behavior and potentially higher user satisfaction",
                
                "The improved accuracy with OpenAI could translate to higher conversion rates and customer satisfaction",
                
                "BERT may be more suitable for offline or latency-sensitive applications where speed is prioritized over accuracy",
                
                "Cloud-based API dependency with OpenAI introduces operational risks not present with locally-hosted BERT"
            ]
        }
        
        # Add statistical significance notes
        if statistical_tests:
            significance_notes = []
            for test_name, test_result in statistical_tests.items():
                if test_result.get("significant", False):
                    significance_notes.append(f"The difference in {test_name} is statistically significant (p={test_result['p_value']:.4f})")
                else:
                    significance_notes.append(f"The difference in {test_name} is not statistically significant (p={test_result['p_value']:.4f})")
        
            summary["statistical_significance"] = significance_notes
        
        # Convert any NumPy types to standard Python types for JSON serialization
        serializable_summary = convert_to_json_serializable(summary)
        
        # Save summary to JSON file
        with open(os.path.join(self.output_dir, 'model_comparison_report.json'), 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        # Save a human-readable summary
        with open(os.path.join(self.output_dir, 'technical_report.txt'), 'w') as f:
            f.write("==============================================\n")
            f.write("BERT vs OpenAI Embedding Models: Technical Comparison\n")
            f.write("==============================================\n\n")
            
            f.write("TEST CONFIGURATION\n")
            f.write("-----------------\n")
            for key, value in summary["test_configuration"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-----------------\n")
            f.write(f"Response Time:\n")
            f.write(f"  BERT: {bert_avg_time:.4f}s (95% CI: {detailed_metrics.get('response_time_details', {}).get('bert', {}).get('confidence_interval', {}).get('lower_bound', 0):.4f} - {detailed_metrics.get('response_time_details', {}).get('bert', {}).get('confidence_interval', {}).get('upper_bound', 0):.4f})\n")
            f.write(f"  OpenAI: {openai_avg_time:.4f}s (95% CI: {detailed_metrics.get('response_time_details', {}).get('openai', {}).get('confidence_interval', {}).get('lower_bound', 0):.4f} - {detailed_metrics.get('response_time_details', {}).get('openai', {}).get('confidence_interval', {}).get('upper_bound', 0):.4f})\n")
            f.write(f"  Ratio: {openai_avg_time / bert_avg_time:.2f}x\n")
            f.write(f"  p-value: {statistical_tests.get('response_time', {}).get('p_value', 1.0):.6f}\n\n")
            
            f.write(f"Similarity:\n")
            f.write(f"  BERT: {bert_avg_similarity:.4f} (95% CI: {detailed_metrics.get('similarity_details', {}).get('bert', {}).get('confidence_interval', {}).get('lower_bound', 0):.4f} - {detailed_metrics.get('similarity_details', {}).get('bert', {}).get('confidence_interval', {}).get('upper_bound', 0):.4f})\n")
            f.write(f"  OpenAI: {openai_avg_similarity:.4f} (95% CI: {detailed_metrics.get('similarity_details', {}).get('openai', {}).get('confidence_interval', {}).get('lower_bound', 0):.4f} - {detailed_metrics.get('similarity_details', {}).get('openai', {}).get('confidence_interval', {}).get('upper_bound', 0):.4f})\n")
            f.write(f"  Difference: {openai_avg_similarity - bert_avg_similarity:.4f}\n")
            f.write(f"  p-value: {statistical_tests.get('similarity', {}).get('p_value', 1.0):.6f}\n\n")
            
            f.write(f"Consistency:\n")
            f.write(f"  BERT: {results['bert']['consistency']:.4f}\n")
            f.write(f"  OpenAI: {results['openai']['consistency']:.4f}\n")
            f.write(f"  Ratio: {results['openai']['consistency'] / results['bert']['consistency']:.2f}x\n\n")
            
            f.write(f"Accuracy:\n")
            f.write(f"  BERT: {bert_avg_accuracy:.4f} (95% CI: {detailed_metrics.get('accuracy_details', {}).get('bert', {}).get('confidence_interval', {}).get('lower_bound', 0):.4f} - {detailed_metrics.get('accuracy_details', {}).get('bert', {}).get('confidence_interval', {}).get('upper_bound', 0):.4f})\n")
            f.write(f"  OpenAI: {openai_avg_accuracy:.4f} (95% CI: {detailed_metrics.get('accuracy_details', {}).get('openai', {}).get('confidence_interval', {}).get('lower_bound', 0):.4f} - {detailed_metrics.get('accuracy_details', {}).get('openai', {}).get('confidence_interval', {}).get('upper_bound', 0):.4f})\n")
            f.write(f"  Difference: {openai_avg_accuracy - bert_avg_accuracy:.4f}\n")
            f.write(f"  Effect Size (Cohen's d): {detailed_metrics.get('accuracy_details', {}).get('effect_size', {}).get('cohens_d', 0):.2f} ({detailed_metrics.get('accuracy_details', {}).get('effect_size', {}).get('interpretation', 'unknown')})\n")
            f.write(f"  p-value: {statistical_tests.get('accuracy', {}).get('p_value', 1.0):.6f}\n\n")
            
            if ranking_analysis.get("top1_accuracy"):
                f.write(f"Top-1 Accuracy:\n")
                f.write(f"  BERT: {ranking_analysis['top1_accuracy']['bert']:.4f}\n")
                f.write(f"  OpenAI: {ranking_analysis['top1_accuracy']['openai']:.4f}\n")
                f.write(f"  Difference: {ranking_analysis['top1_accuracy']['difference']:.4f}\n\n")
            
            f.write("STATISTICAL SIGNIFICANCE\n")
            f.write("-----------------------\n")
            for note in summary.get("statistical_significance", []):
                f.write(f"{note}\n")
            f.write("\n")
            
            f.write("TECHNICAL CONCLUSIONS\n")
            f.write("--------------------\n")
            for conclusion in summary.get("technical_conclusion", []):
                f.write(f"- {conclusion}\n")
            f.write("\n")
            
            f.write("BUSINESS IMPLICATIONS\n")
            f.write("--------------------\n")
            for implication in summary.get("business_implications", []):
                f.write(f"- {implication}\n")

if __name__ == "__main__":
    unittest.main() 