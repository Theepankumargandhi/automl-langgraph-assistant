# cost_tracker.py
from typing import Dict, Any
import time

class APIUsageTracker:
    def __init__(self):
        self.session_costs = []
        self.current_run_costs = []
        
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
            return cost_info
            
        except Exception as e:
            print(f"Cost tracking error: {e}")
            return {'operation': operation, 'cost_usd': 0.0, 'input_tokens': 0, 'output_tokens': 0}
    
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
        if self.current_run_costs:
            self.session_costs.append(self.current_run_costs.copy())
        self.current_run_costs = []
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by operation"""
        breakdown = {}
        for call in self.current_run_costs:
            op = call['operation']
            breakdown[op] = breakdown.get(op, 0) + call['cost_usd']
        return breakdown