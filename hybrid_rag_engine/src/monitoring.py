"""
Performance Monitoring & Analytics for Advanced Hybrid RAG Engine

This module provides comprehensive performance monitoring, metrics collection,
and system analytics for production RAG deployments.

Features:
- Real-time performance metrics
- Query latency analysis  
- Resource utilization monitoring
- Cache performance tracking
- LLM performance comparison
- Custom alerting thresholds
"""

import time
import psutil
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Individual query performance metrics"""
    query_id: str
    timestamp: datetime
    question_length: int
    processing_time: float
    llm_provider: str
    model: str
    chunks_retrieved: int
    chunks_used: int
    response_length: int
    cache_hit: bool
    semantic_similarity_score: float = 0.0
    error_occurred: bool = False
    error_type: Optional[str] = None

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    active_connections: int
    
@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    timestamp: datetime
    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_size: int
    evictions: int
    hit_rate: float

class PerformanceMonitor:
    """Real-time performance monitoring and analytics"""
    
    def __init__(self, retention_hours: int = 24, metrics_file: str = "metrics.jsonl"):
        self.retention_hours = retention_hours
        self.metrics_file = Path(metrics_file)
        
        # In-memory metrics storage
        self.query_metrics: deque = deque(maxlen=10000)
        self.system_metrics: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.cache_metrics: deque = deque(maxlen=1440)
        
        # Aggregated statistics
        self.hourly_stats = defaultdict(lambda: {
            'query_count': 0,
            'avg_processing_time': 0.0,
            'error_rate': 0.0,
            'cache_hit_rate': 0.0
        })
        
        # Performance thresholds for alerting
        self.thresholds = {
            'max_processing_time': 10.0,  # seconds
            'max_error_rate': 0.05,       # 5%
            'min_cache_hit_rate': 0.2,    # 20%
            'max_cpu_usage': 0.8,         # 80%
            'max_memory_usage': 0.8       # 80%
        }
        
        # Background monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background system monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitor_system(self):
        """Background system monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metric = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_usage=psutil.cpu_percent(interval=1),
                    memory_usage=psutil.virtual_memory().percent / 100,
                    memory_available=psutil.virtual_memory().available / (1024**3),  # GB
                    disk_usage=psutil.disk_usage('/').percent / 100,
                    active_connections=len(psutil.net_connections())
                )
                
                self.system_metrics.append(system_metric)
                
                # Check thresholds and alert if needed
                self._check_thresholds(system_metric)
                
                # Cleanup old metrics
                self._cleanup_metrics()
                
                # Save metrics to disk periodically
                if len(self.system_metrics) % 10 == 0:
                    self._save_metrics()
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
            
            time.sleep(60)  # Monitor every minute
    
    def record_query(self, metrics: QueryMetrics):
        """Record query performance metrics"""
        self.query_metrics.append(metrics)
        
        # Update hourly statistics
        hour_key = metrics.timestamp.strftime('%Y-%m-%d %H')
        stats = self.hourly_stats[hour_key]
        
        stats['query_count'] += 1
        
        # Update running average for processing time
        current_avg = stats['avg_processing_time']
        count = stats['query_count']
        stats['avg_processing_time'] = (current_avg * (count - 1) + metrics.processing_time) / count
        
        # Update error rate
        if metrics.error_occurred:
            stats['error_count'] = stats.get('error_count', 0) + 1
        stats['error_rate'] = stats.get('error_count', 0) / count
        
        # Update cache hit rate
        if metrics.cache_hit:
            stats['cache_hits'] = stats.get('cache_hits', 0) + 1
        stats['cache_hit_rate'] = stats.get('cache_hits', 0) / count
        
        logger.debug(f"Recorded query metrics: {metrics.query_id}")
    
    def record_cache_metrics(self, total_requests: int, cache_hits: int, 
                           cache_size: int, evictions: int = 0):
        """Record cache performance metrics"""
        cache_misses = total_requests - cache_hits
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        
        cache_metric = CacheMetrics(
            timestamp=datetime.now(),
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_size=cache_size,
            evictions=evictions,
            hit_rate=hit_rate
        )
        
        self.cache_metrics.append(cache_metric)
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent query metrics
        recent_queries = [m for m in self.query_metrics if m.timestamp >= cutoff_time]
        
        if not recent_queries:
            return {"error": "No queries in the specified time period"}
        
        # Calculate aggregate metrics
        total_queries = len(recent_queries)
        avg_processing_time = np.mean([q.processing_time for q in recent_queries])
        median_processing_time = np.median([q.processing_time for q in recent_queries])
        p95_processing_time = np.percentile([q.processing_time for q in recent_queries], 95)
        
        error_count = sum(1 for q in recent_queries if q.error_occurred)
        error_rate = error_count / total_queries
        
        cache_hits = sum(1 for q in recent_queries if q.cache_hit)
        cache_hit_rate = cache_hits / total_queries
        
        # LLM provider breakdown
        provider_stats = defaultdict(lambda: {'count': 0, 'avg_time': 0.0})
        for query in recent_queries:
            provider = query.llm_provider
            provider_stats[provider]['count'] += 1
            current_avg = provider_stats[provider]['avg_time']
            count = provider_stats[provider]['count']
            provider_stats[provider]['avg_time'] = (
                (current_avg * (count - 1) + query.processing_time) / count
            )
        
        # System resource summary
        recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        if recent_system:
            avg_cpu = np.mean([s.cpu_usage for s in recent_system])
            avg_memory = np.mean([s.memory_usage for s in recent_system])
            min_memory_available = min([s.memory_available for s in recent_system])
        else:
            avg_cpu = avg_memory = min_memory_available = 0.0
        
        return {
            "time_period": f"Last {hours} hour(s)",
            "query_metrics": {
                "total_queries": total_queries,
                "avg_processing_time": round(avg_processing_time, 3),
                "median_processing_time": round(median_processing_time, 3),
                "p95_processing_time": round(p95_processing_time, 3),
                "error_rate": round(error_rate, 3),
                "cache_hit_rate": round(cache_hit_rate, 3)
            },
            "system_metrics": {
                "avg_cpu_usage": round(avg_cpu, 3),
                "avg_memory_usage": round(avg_memory, 3),
                "min_memory_available_gb": round(min_memory_available, 2)
            },
            "llm_provider_stats": dict(provider_stats),
            "performance_alerts": self._get_active_alerts()
        }
    
    def get_detailed_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics and insights"""
        all_queries = list(self.query_metrics)
        
        if not all_queries:
            return {"error": "No query data available"}
        
        # Time-based analysis
        hourly_analysis = self._analyze_by_time_period(all_queries, 'hour')
        daily_analysis = self._analyze_by_time_period(all_queries, 'day')
        
        # Query complexity analysis
        complexity_analysis = self._analyze_query_complexity(all_queries)
        
        # Performance trends
        trends = self._calculate_performance_trends(all_queries)
        
        # Model comparison
        model_comparison = self._compare_models(all_queries)
        
        return {
            "overview": {
                "total_queries": len(all_queries),
                "date_range": {
                    "start": min(q.timestamp for q in all_queries).isoformat(),
                    "end": max(q.timestamp for q in all_queries).isoformat()
                }
            },
            "hourly_analysis": hourly_analysis,
            "daily_analysis": daily_analysis,
            "complexity_analysis": complexity_analysis,
            "performance_trends": trends,
            "model_comparison": model_comparison,
            "recommendations": self._generate_recommendations(all_queries)
        }
    
    def _analyze_by_time_period(self, queries: List[QueryMetrics], period: str) -> Dict:
        """Analyze performance by time period"""
        period_stats = defaultdict(lambda: {
            'query_count': 0,
            'total_time': 0.0,
            'error_count': 0,
            'cache_hits': 0
        })
        
        for query in queries:
            if period == 'hour':
                key = query.timestamp.strftime('%Y-%m-%d %H:00')
            else:  # day
                key = query.timestamp.strftime('%Y-%m-%d')
            
            stats = period_stats[key]
            stats['query_count'] += 1
            stats['total_time'] += query.processing_time
            
            if query.error_occurred:
                stats['error_count'] += 1
            if query.cache_hit:
                stats['cache_hits'] += 1
        
        # Calculate derived metrics
        for stats in period_stats.values():
            count = stats['query_count']
            if count > 0:
                stats['avg_processing_time'] = stats['total_time'] / count
                stats['error_rate'] = stats['error_count'] / count
                stats['cache_hit_rate'] = stats['cache_hits'] / count
        
        return dict(period_stats)
    
    def _analyze_query_complexity(self, queries: List[QueryMetrics]) -> Dict:
        """Analyze performance by query complexity"""
        complexity_bins = {
            'short': {'min': 0, 'max': 50, 'queries': []},
            'medium': {'min': 50, 'max': 200, 'queries': []},
            'long': {'min': 200, 'max': float('inf'), 'queries': []}
        }
        
        for query in queries:
            length = query.question_length
            for bin_name, bin_info in complexity_bins.items():
                if bin_info['min'] <= length < bin_info['max']:
                    bin_info['queries'].append(query)
                    break
        
        analysis = {}
        for bin_name, bin_info in complexity_bins.items():
            query_list = bin_info['queries']
            if query_list:
                analysis[bin_name] = {
                    'count': len(query_list),
                    'avg_processing_time': np.mean([q.processing_time for q in query_list]),
                    'avg_length': np.mean([q.question_length for q in query_list])
                }
        
        return analysis
    
    def _calculate_performance_trends(self, queries: List[QueryMetrics]) -> Dict:
        """Calculate performance trends over time"""
        if len(queries) < 10:
            return {"error": "Insufficient data for trend analysis"}
        
        # Sort by timestamp
        sorted_queries = sorted(queries, key=lambda x: x.timestamp)
        
        # Calculate moving averages
        window_size = min(20, len(sorted_queries) // 2)
        processing_times = [q.processing_time for q in sorted_queries]
        
        moving_avg = []
        for i in range(window_size, len(processing_times)):
            avg = np.mean(processing_times[i-window_size:i])
            moving_avg.append(avg)
        
        # Calculate trend (simple linear regression slope)
        if len(moving_avg) > 1:
            x = np.arange(len(moving_avg))
            slope = np.polyfit(x, moving_avg, 1)[0]
            trend = "improving" if slope < -0.01 else "degrading" if slope > 0.01 else "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "slope": slope if len(moving_avg) > 1 else 0.0,
            "recent_avg": np.mean(processing_times[-10:]) if len(processing_times) >= 10 else 0.0,
            "baseline_avg": np.mean(processing_times[:10]) if len(processing_times) >= 10 else 0.0
        }
    
    def _compare_models(self, queries: List[QueryMetrics]) -> Dict:
        """Compare performance across different models"""
        model_stats = defaultdict(lambda: {
            'queries': [],
            'total_time': 0.0,
            'error_count': 0
        })
        
        for query in queries:
            key = f"{query.llm_provider}:{query.model}"
            stats = model_stats[key]
            stats['queries'].append(query)
            stats['total_time'] += query.processing_time
            
            if query.error_occurred:
                stats['error_count'] += 1
        
        comparison = {}
        for model_key, stats in model_stats.items():
            query_list = stats['queries']
            count = len(query_list)
            
            if count > 0:
                comparison[model_key] = {
                    'query_count': count,
                    'avg_processing_time': stats['total_time'] / count,
                    'error_rate': stats['error_count'] / count,
                    'median_processing_time': np.median([q.processing_time for q in query_list])
                }
        
        return comparison
    
    def _generate_recommendations(self, queries: List[QueryMetrics]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not queries:
            return recommendations
        
        # Analyze cache performance
        cache_hit_rate = sum(1 for q in queries if q.cache_hit) / len(queries)
        if cache_hit_rate < 0.3:
            recommendations.append("Consider increasing cache size or TTL to improve cache hit rate")
        
        # Analyze processing times
        avg_time = np.mean([q.processing_time for q in queries])
        if avg_time > 5.0:
            recommendations.append("Average processing time is high - consider optimizing retrieval or using faster models")
        
        # Analyze error rates
        error_rate = sum(1 for q in queries if q.error_occurred) / len(queries)
        if error_rate > 0.05:
            recommendations.append("Error rate is elevated - investigate common failure patterns")
        
        # Model-specific recommendations
        model_comparison = self._compare_models(queries)
        if len(model_comparison) > 1:
            fastest_model = min(model_comparison.items(), key=lambda x: x[1]['avg_processing_time'])
            recommendations.append(f"Consider using {fastest_model[0]} for better performance")
        
        return recommendations
    
    def _check_thresholds(self, system_metric: SystemMetrics):
        """Check performance thresholds and generate alerts"""
        alerts = []
        
        if system_metric.cpu_usage > self.thresholds['max_cpu_usage']:
            alerts.append(f"High CPU usage: {system_metric.cpu_usage:.1%}")
        
        if system_metric.memory_usage > self.thresholds['max_memory_usage']:
            alerts.append(f"High memory usage: {system_metric.memory_usage:.1%}")
        
        if system_metric.memory_available < 1.0:  # Less than 1GB
            alerts.append(f"Low memory available: {system_metric.memory_available:.1f}GB")
        
        if alerts:
            logger.warning(f"Performance alerts: {', '.join(alerts)}")
    
    def _get_active_alerts(self) -> List[str]:
        """Get current active performance alerts"""
        alerts = []
        
        # Check recent system metrics
        if self.system_metrics:
            latest = self.system_metrics[-1]
            
            if latest.cpu_usage > self.thresholds['max_cpu_usage']:
                alerts.append(f"High CPU usage: {latest.cpu_usage:.1%}")
            
            if latest.memory_usage > self.thresholds['max_memory_usage']:
                alerts.append(f"High memory usage: {latest.memory_usage:.1%}")
        
        # Check recent query performance
        recent_queries = [q for q in self.query_metrics 
                         if q.timestamp > datetime.now() - timedelta(minutes=5)]
        
        if recent_queries:
            avg_time = np.mean([q.processing_time for q in recent_queries])
            if avg_time > self.thresholds['max_processing_time']:
                alerts.append(f"Slow queries: {avg_time:.2f}s average")
        
        return alerts
    
    def _cleanup_metrics(self):
        """Clean up old metrics based on retention policy"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Clean query metrics
        self.query_metrics = deque(
            [m for m in self.query_metrics if m.timestamp >= cutoff_time],
            maxlen=self.query_metrics.maxlen
        )
        
        # Clean hourly stats
        hours_to_keep = set()
        for i in range(self.retention_hours + 1):
            hour_key = (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H')
            hours_to_keep.add(hour_key)
        
        keys_to_remove = [k for k in self.hourly_stats.keys() if k not in hours_to_keep]
        for key in keys_to_remove:
            del self.hourly_stats[key]
    
    def _save_metrics(self):
        """Save metrics to disk for persistence"""
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.metrics_file, 'w') as f:
                # Save recent query metrics
                for metric in list(self.query_metrics)[-100:]:  # Last 100 queries
                    json.dump({
                        'type': 'query',
                        'timestamp': metric.timestamp.isoformat(),
                        'data': asdict(metric)
                    }, f)
                    f.write('\n')
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def create_query_metrics(query_id: str, question: str, processing_time: float,
                        llm_provider: str, model: str, chunks_retrieved: int,
                        chunks_used: int, response: str, cache_hit: bool = False,
                        error_occurred: bool = False, error_type: str = None) -> QueryMetrics:
    """Helper function to create QueryMetrics object"""
    return QueryMetrics(
        query_id=query_id,
        timestamp=datetime.now(),
        question_length=len(question),
        processing_time=processing_time,
        llm_provider=llm_provider,
        model=model,
        chunks_retrieved=chunks_retrieved,
        chunks_used=chunks_used,
        response_length=len(response),
        cache_hit=cache_hit,
        error_occurred=error_occurred,
        error_type=error_type
    )