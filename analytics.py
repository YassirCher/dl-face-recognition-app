import json
import os

class Analytics:
    def __init__(self, data_paths):
        """
        Args:
            data_paths: A single file path (str) or a list of file paths
                        to model results JSON files.
        """
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.data_paths = data_paths
        self.results = self._load_data()

    def _load_data(self):
        all_results = []
        for path in self.data_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_results.extend(data)
        return all_results

    def get_training_health(self):
        """
        Calculate Overfitting and Stability scores for each model.
        """
        health_stats = []
        for r in self.results:
            name = r['model_name']
            bt = r.get('backbone_training', {})
            
            # Overfitting Score: Difference between final Train and Val accuracy
            train_acc = bt.get('train_accuracy', [])
            val_acc = bt.get('val_accuracy', [])
            overfitting_score = 0
            if train_acc and val_acc:
                overfitting_score = train_acc[-1] - val_acc[-1]

            # Stability Score: Std Dev of Validation Loss over last 3 epochs
            val_loss = bt.get('val_loss', [])
            stability_score = 0
            if len(val_loss) >= 3:
                recent_loss = val_loss[-3:]
                mean_loss = sum(recent_loss) / len(recent_loss)
                variance = sum([((x - mean_loss) ** 2) for x in recent_loss]) / len(recent_loss)
                stability_score = variance ** 0.5
            elif val_loss:
                 stability_score = 0 # Not enough data, assume stable or handle otherwise

            health_stats.append({
                'model_name': name,
                'overfitting_score': round(overfitting_score, 2),
                'stability_score': round(stability_score, 4),
                'is_overfitting': overfitting_score > 5.0, # Threshold assumption
                'is_unstable': stability_score > 0.1 # Threshold assumption
            })
        return health_stats

    def get_analytics_data(self):
        """
        Returns data formatted for Chart.js.
        """
        if not self.results:
            return {}

        # Basic lists
        model_names = [r['model_name'] for r in self.results]
        accuracies = [r.get('classifier_metrics', {}).get('test_accuracy', 0) for r in self.results]
        f1_scores = [r.get('classifier_metrics', {}).get('f1', 0) for r in self.results]
        training_times = [r.get('training_time_seconds', 0) for r in self.results]
        top5_accuracies = [r.get('classifier_metrics', {}).get('top5_accuracy', 0) for r in self.results]
        auc_rocs = [r.get('classifier_metrics', {}).get('auc_roc', 0) for r in self.results]

        # Health Stats
        health_stats = self.get_training_health()

        # Advanced Aggregations
        # Initialize containers
        metrics_by_backbone = {}
        metrics_by_classifier = {}
        
        # Helper to initialize dict
        def init_metrics():
            return {'accuracy': [], 'f1': [], 'auc': [], 'time': [], 'count': 0}

        for r in self.results:
            bb = r['backbone']
            cl = r['classifier']
            
            # Extract Metrics
            acc = r.get('classifier_metrics', {}).get('test_accuracy', 0)
            f1 = r.get('classifier_metrics', {}).get('f1', 0)
            auc = r.get('classifier_metrics', {}).get('auc_roc', 0)
            time = r.get('training_time_seconds', 0)

            # Aggregate by Backbone
            if bb not in metrics_by_backbone: metrics_by_backbone[bb] = init_metrics()
            metrics_by_backbone[bb]['accuracy'].append(acc)
            metrics_by_backbone[bb]['f1'].append(f1)
            metrics_by_backbone[bb]['auc'].append(auc)
            metrics_by_backbone[bb]['time'].append(time)
            metrics_by_backbone[bb]['count'] += 1

            # Aggregate by Classifier
            if cl not in metrics_by_classifier: metrics_by_classifier[cl] = init_metrics()
            metrics_by_classifier[cl]['accuracy'].append(acc)
            metrics_by_classifier[cl]['f1'].append(f1)
            metrics_by_classifier[cl]['auc'].append(auc)
            metrics_by_classifier[cl]['time'].append(time)
            metrics_by_classifier[cl]['count'] += 1

        # Calculate Averages
        def calculate_means(metric_dict):
            means = {}
            for key, values in metric_dict.items():
                if values['count'] > 0:
                    means[key] = {
                        'avg_accuracy': sum(values['accuracy']) / values['count'],
                        'avg_f1': sum(values['f1']) / values['count'],
                        'avg_auc': sum(values['auc']) / values['count'],
                        'avg_time': sum(values['time']) / values['count']
                    }
                else:
                    means[key] = {'avg_accuracy': 0, 'avg_f1': 0, 'avg_auc': 0, 'avg_time': 0}
            return means

        backbone_means = calculate_means(metrics_by_backbone)
        classifier_means = calculate_means(metrics_by_classifier)

        # Simple key-value for pie charts (legacy compatibility + new)
        backbone_performance = {k: v['avg_accuracy'] for k, v in backbone_means.items()}
        classifier_performance = {k: v['avg_accuracy'] for k, v in classifier_means.items()}

        # ── NEW: Neural vs Classical comparison ──
        neural_vs_classical = {}
        # Helper to initialize dict
        def init_metrics():
            return {'accuracy': [], 'f1': [], 'auc': [], 'time': [], 'count': 0}

        for r in self.results:
            ctype = r.get('classifier_type', 'unknown')
            if ctype not in neural_vs_classical:
                neural_vs_classical[ctype] = init_metrics()
            cm = r.get('classifier_metrics', {})
            neural_vs_classical[ctype]['accuracy'].append(cm.get('test_accuracy', 0))
            neural_vs_classical[ctype]['f1'].append(cm.get('f1', 0))
            neural_vs_classical[ctype]['auc'].append(cm.get('auc_roc', 0))
            neural_vs_classical[ctype]['time'].append(r.get('training_time_seconds', 0))
            neural_vs_classical[ctype]['count'] += 1
        neural_vs_classical_means = calculate_means(neural_vs_classical)

        # ── NEW: Convergence analysis (neural models only) ──
        convergence_analysis = []
        for r in self.results:
            bt = r.get('backbone_training', {})
            val_acc = bt.get('val_accuracy', [])
            val_loss = bt.get('val_loss', [])
            if not val_acc or len(val_acc) < 2:
                continue  # skip classical models with no epoch data
            best_epoch = int(val_acc.index(max(val_acc)) + 1)
            improvement_rate = round((val_acc[-1] - val_acc[0]) / len(val_acc), 2)
            plateau_delta = round(abs(val_loss[-1] - val_loss[-2]), 4) if len(val_loss) >= 2 else 0
            convergence_analysis.append({
                'model_name': r['model_name'],
                'best_epoch': best_epoch,
                'total_epochs': r.get('epochs', len(val_acc)),
                'improvement_rate': improvement_rate,
                'plateau_delta': plateau_delta,
                'final_val_acc': round(val_acc[-1], 2)
            })

        # ── NEW: Efficiency scores (accuracy per minute) ──
        efficiency_scores = []
        for i, r in enumerate(self.results):
            time_min = r.get('training_time_seconds', 1) / 60.0
            acc = r.get('classifier_metrics', {}).get('test_accuracy', 0)
            f1_val = r.get('classifier_metrics', {}).get('f1', 0)
            efficiency_scores.append({
                'model_name': r['model_name'],
                'accuracy': round(acc, 2),
                'f1': round(f1_val, 4),
                'time_minutes': round(time_min, 1),
                'acc_per_minute': round(acc / max(time_min, 0.1), 2)
            })

        # ── NEW: Accuracy vs F1 scatter data ──
        accuracy_vs_f1_scatter = []
        for r in self.results:
            cm = r.get('classifier_metrics', {})
            accuracy_vs_f1_scatter.append({
                'model_name': r['model_name'],
                'accuracy': round(cm.get('test_accuracy', 0), 2),
                'f1': round(cm.get('f1', 0), 4)
            })

        # ── NEW: Model rankings (Existing logic, kept for compatibility) ──
        n = len(self.results)
        sorted_by_acc = sorted(range(n), key=lambda i: accuracies[i], reverse=True)
        sorted_by_f1 = sorted(range(n), key=lambda i: f1_scores[i], reverse=True)
        sorted_by_auc = sorted(range(n), key=lambda i: auc_rocs[i], reverse=True)
        sorted_by_time = sorted(range(n), key=lambda i: training_times[i])  # lower is better

        rank_acc = [0] * n
        rank_f1 = [0] * n
        rank_auc = [0] * n
        rank_time = [0] * n
        for rank, idx in enumerate(sorted_by_acc): rank_acc[idx] = rank + 1
        for rank, idx in enumerate(sorted_by_f1): rank_f1[idx] = rank + 1
        for rank, idx in enumerate(sorted_by_auc): rank_auc[idx] = rank + 1
        for rank, idx in enumerate(sorted_by_time): rank_time[idx] = rank + 1

        model_rankings = []
        for i, r in enumerate(self.results):
            avg_rank = (rank_acc[i] + rank_f1[i] + rank_auc[i] + rank_time[i]) / 4.0
            model_rankings.append({
                'model_name': r['model_name'],
                'rank_acc': rank_acc[i],
                'rank_f1': rank_f1[i],
                'rank_auc': rank_auc[i],
                'rank_time': rank_time[i],
                'avg_rank': round(avg_rank, 1)
            })
        model_rankings.sort(key=lambda x: x['avg_rank'])

        # ── NEW: Leaderboard Data (Structured for DataTables) ──
        leaderboard = []
        for r in self.results:
            leaderboard.append({
                'name': r['model_name'],
                'backbone': r['backbone'],
                'classifier': r['classifier'],
                'accuracy': round(r.get('classifier_metrics', {}).get('test_accuracy', 0), 2),
                'f1': round(r.get('classifier_metrics', {}).get('f1', 0), 4),
                'time': round(r.get('training_time_seconds', 0), 2)
            })
        # Sort by accuracy descending initially
        leaderboard.sort(key=lambda x: x['accuracy'], reverse=True)
        for i, item in enumerate(leaderboard):
            item['rank'] = i + 1

        # ── NEW: Bubble Chart Data (Efficiency Visualization) ──
        # x=time, y=accuracy, r=f1 (scaled for visibility)
        efficiency_bubble = []
        for r in self.results:
            f1 = r.get('classifier_metrics', {}).get('f1', 0)
            efficiency_bubble.append({
                'x': round(r.get('training_time_seconds', 0), 2),
                'y': round(r.get('classifier_metrics', {}).get('test_accuracy', 0), 2),
                'r': round(f1 * 15, 2), # Scale factor 15 for visible bubbles
                'model': r['model_name'],
                'backbone': r['backbone']
            })
            
        # ── NEW: Backbone Comparison (Box Plot / Distribution Data) ──
        # We compute min, q1, median, q3, max for accuracy per backbone
        import statistics
        backbone_distribution = {}
        # First, build raw data with classifier names
        bb_raw_data = {}
        for r in self.results:
            bb = r['backbone']
            cl = r['classifier']
            acc = r.get('classifier_metrics', {}).get('test_accuracy', 0)
            if bb not in bb_raw_data:
                bb_raw_data[bb] = []
            bb_raw_data[bb].append({'acc': acc, 'classifier': cl})

        for bb, items in bb_raw_data.items():
            accs = sorted([d['acc'] for d in items])
            if not accs: continue
            
            # Simple quartiles
            n_accs = len(accs)
            q1_idx = int(n_accs * 0.25)
            q3_idx = int(n_accs * 0.75)
            
            backbone_distribution[bb] = {
                'min': min(accs),
                'q1': accs[q1_idx],
                'median': statistics.median(accs),
                'q3': accs[q3_idx],
                'max': max(accs),
                'raw': [{'acc': d['acc'], 'classifier': d['classifier']} for d in sorted(items, key=lambda x: x['acc'])]
            }

        # ── NEW: Heatmap Data (Backbone vs Classifier Performance) ──
        # Structure: x=Classifier, y=Backbone, v=Accuracy
        heatmap_data = []
        for r in self.results:
            heatmap_data.append({
                'x': r['classifier'],
                'y': r['backbone'],
                'v': round(r.get('classifier_metrics', {}).get('test_accuracy', 0), 2)
            })

        return {
            'model_names': model_names,
            'accuracies': accuracies,
            'f1_scores': f1_scores,
            'training_times': training_times,
            'top5_accuracies': top5_accuracies,
            'auc_rocs': auc_rocs,

            'health_stats': health_stats,

            'backbone_means': backbone_means,
            'classifier_means': classifier_means,

            'backbone_performance': backbone_performance,
            'classifier_performance': classifier_performance,

            # New analysis sections
            'neural_vs_classical': neural_vs_classical_means, # Use the computed means!
            'convergence_analysis': convergence_analysis,
            'efficiency_scores': efficiency_scores,
            'accuracy_vs_f1_scatter': accuracy_vs_f1_scatter,
            'model_rankings': model_rankings,
            
            # New Advanced Charts
            'leaderboard': leaderboard,
            'efficiency_bubble': efficiency_bubble,
            'backbone_distribution': backbone_distribution,
            'heatmap_data': heatmap_data,

            'raw_data': self.results
        }
