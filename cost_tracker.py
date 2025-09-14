# cost_tracker.py
from typing import Dict, Any
import time

class APIUsageTracker:
    def __init__(self):
        self.session_costs = []
        self.current_run_costs = []
        self.mlflow_enabled = False
        self.logged_params = set()  # Track logged parameters to avoid duplicates
        
    def enable_mlflow_logging(self):
        """Enable MLflow logging for cost tracking"""
        try:
            import mlflow
            self.mlflow_enabled = True
        except ImportError:
            self.mlflow_enabled = False
        
    def track_openai_call(self, response: Any, operation: str = "unknown"):
        """Track OpenAI API usage from response object"""
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
            elif hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                usage = response.response_metadata['token_usage']
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
            else:
                # Fallback - estimate tokens (rough approximation)
                content = getattr(response, 'content', str(response))
                total_chars = len(content)
                input_tokens = total_chars // 4  # rough estimate
                output_tokens = total_chars // 4
                
            # GPT-4o-mini pricing (as of 2024)
            input_cost = (input_tokens / 1000) * 0.00015  # $0.15 per 1K input tokens
            output_cost = (output_tokens / 1000) * 0.0006  # $0.60 per 1K output tokens
            total_cost = input_cost + output_cost
            
            cost_info = {
                'operation': operation,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost_usd': round(total_cost, 4),
                'timestamp': time.time()
            }
            
            self.current_run_costs.append(cost_info)
            
            # Log to MLflow if enabled
            if self.mlflow_enabled:
                self._log_to_mlflow(cost_info)
            
            return cost_info
            
        except Exception as e:
            print(f"Cost tracking error: {e}")
            return {'operation': operation, 'cost_usd': 0.0, 'input_tokens': 0, 'output_tokens': 0}
    
    def _log_to_mlflow(self, cost_info: Dict[str, Any]):
        """Log individual API call to MLflow"""
        try:
            import mlflow
            
            # Safely truncate operation name for metric keys
            safe_op = self._sanitize_key(cost_info['operation'])
            
            # Log token usage metrics
            mlflow.log_metric(f"tokens.{safe_op}.input", cost_info['input_tokens'])
            mlflow.log_metric(f"tokens.{safe_op}.output", cost_info['output_tokens'])
            mlflow.log_metric(f"cost.{safe_op}.usd", cost_info['cost_usd'])
            
        except Exception as e:
            print(f"MLflow cost logging error: {e}")
    
    def _sanitize_key(self, key: str, max_length: int = 40) -> str:
        """Sanitize key for MLflow parameter/metric names"""
        # Remove invalid characters and truncate
        import re
        # Keep only alphanumerics, underscores, dashes, periods, and spaces
        sanitized = re.sub(r'[^a-zA-Z0-9_\-\. ]', '', str(key))
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length-3] + "..."
        return sanitized
    
    def _safe_log_param(self, key: str, value: str):
        """Safely log parameter, avoiding duplicates"""
        try:
            import mlflow
            safe_key = self._sanitize_key(key)
            
            # Skip if already logged in this run
            if safe_key in self.logged_params:
                return
                
            mlflow.log_param(safe_key, value)
            self.logged_params.add(safe_key)
            
        except Exception as e:
            if "already logged" not in str(e):
                print(f"MLflow param logging error: {e}")
    
    def get_current_run_total(self) -> float:
        """Get total cost for current pipeline run"""
        return sum(call['cost_usd'] for call in self.current_run_costs)
    
    def get_session_total(self) -> float:
        """Get total cost for entire session"""
        current_total = self.get_current_run_total()
        session_total = sum(sum(call['cost_usd'] for call in run) for run in self.session_costs)
        return current_total + session_total
    
    def finish_run(self):
        """Mark current run as complete and start new run"""
        # Log final cost summary to MLflow
        if self.mlflow_enabled and self.current_run_costs:
            self._log_final_cost_summary()
        
        if self.current_run_costs:
            self.session_costs.append(self.current_run_costs.copy())
        self.current_run_costs = []
        self.logged_params.clear()  # Reset for next run
    
    def _log_final_cost_summary(self):
        """Log final cost summary metrics to MLflow"""
        try:
            import mlflow
            
            total_cost = self.get_current_run_total()
            breakdown = self.get_breakdown()
            
            # Log total costs (metrics are safe to overwrite)
            mlflow.log_metric("cost.total_usd", total_cost)
            mlflow.log_metric("cost.total_calls", len(self.current_run_costs))
            
            # Log total tokens
            total_input_tokens = sum(call['input_tokens'] for call in self.current_run_costs)
            total_output_tokens = sum(call['output_tokens'] for call in self.current_run_costs)
            mlflow.log_metric("tokens.total_input", total_input_tokens)
            mlflow.log_metric("tokens.total_output", total_output_tokens)
            mlflow.log_metric("tokens.total", total_input_tokens + total_output_tokens)
            
            # Log cost efficiency metrics
            if total_input_tokens + total_output_tokens > 0:
                cost_per_token = total_cost / (total_input_tokens + total_output_tokens)
                mlflow.log_metric("cost.per_token_usd", cost_per_token)
            
            # Log operation breakdown as parameters (with safe key handling)
            for operation, cost in breakdown.items():
                safe_key = f"cost_breakdown.{self._sanitize_key(operation)}"
                self._safe_log_param(safe_key, f"${cost:.4f}")
                
        except Exception as e:
            print(f"MLflow final cost logging error: {e}")
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by operation"""
        breakdown = {}
        for call in self.current_run_costs:
            op = call['operation']
            breakdown[op] = breakdown.get(op, 0) + call['cost_usd']
        return breakdown
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for MLflow logging"""
        if not self.current_run_costs:
            return {}
        
        total_cost = self.get_current_run_total()
        breakdown = self.get_breakdown()
        
        # Calculate token statistics
        total_input_tokens = sum(call['input_tokens'] for call in self.current_run_costs)
        total_output_tokens = sum(call['output_tokens'] for call in self.current_run_costs)
        total_tokens = total_input_tokens + total_output_tokens
        
        # Calculate timing statistics
        timestamps = [call['timestamp'] for call in self.current_run_costs]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        
        return {
            'total_cost_usd': total_cost,
            'total_calls': len(self.current_run_costs),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'cost_per_token': total_cost / total_tokens if total_tokens > 0 else 0,
            'duration_seconds': duration,
            'calls_per_second': len(self.current_run_costs) / duration if duration > 0 else 0,
            'cost_breakdown': breakdown
        }