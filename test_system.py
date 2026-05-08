import json
import time
import sys
import httpx
from typing import Optional

# Configuration
BASE_URL = "http://localhost:8000"
QUERY_URL = f"{BASE_URL}/api/v1/query"
HEALTH_URL = f"{BASE_URL}/health"
EVALS_URL = f"{BASE_URL}/api/v1/evals/latest"

# ANSI Terminal Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_success(msg): print(f"{Colors.GREEN}[PASS]{Colors.ENDC}  {msg}")
def log_fail(msg): print(f"{Colors.RED}[FAIL]{Colors.ENDC}  {msg}")
def log_info(msg): print(f"{Colors.CYAN}  >>>  {Colors.ENDC}{msg}")
def log_agent(agent, status, budget=None):
    budget_str = f"  budget_remaining={budget}" if budget else ""
    print(f"       {Colors.YELLOW}agent='{agent}'{Colors.ENDC}  status='{status}'{budget_str}")

def run_tests():
    print(f"\n{Colors.BOLD}{Colors.HEADER}Real-Time Multi-Agent LLM Orchestration System  —  Integration Test Suite{Colors.ENDC}")
    print(f"{Colors.BOLD}Target: {BASE_URL}{Colors.ENDC}\n")

    # ---------------------------------------------------------
    # 1. HEALTH CHECK
    # ---------------------------------------------------------
    print(f"{'-'*64}\n  TEST 0 · Health Check  GET /health\n{'-'*64}")
    try:
        response = httpx.get(HEALTH_URL, timeout=10.0)
        if response.status_code == 200:
            log_success(f"Server is healthy  (HTTP 200)")
        else:
            log_fail(f"Server health check failed with status {response.status_code}")
            return
    except Exception as e:
        log_fail(f"Could not connect to server: {e}")
        return

    # ---------------------------------------------------------
    # 2. MAIN PIPELINE (SSE Streaming)
    # ---------------------------------------------------------
    print(f"\n{'-'*64}\n  TEST 1 · Main Pipeline  POST /api/v1/query  (SSE stream)\n{'-'*64}")
    
    query_payload = {
        "query": "What is the capital of France, and can you write a small Python script to calculate the area of a circle?",
        "user_id": "qa_tester_001"
    }
    
    job_id: Optional[str] = None
    collected_tokens = []
    activity_count = 0

    log_info(f"Query: '{query_payload['query']}'")
    
    try:
        with httpx.stream("POST", QUERY_URL, json=query_payload, timeout=120.0) as response:
            if response.status_code != 200:
                log_fail(f"Query endpoint failed with status {response.status_code}")
                print(response.text)
                return
            
            log_success(f"Connection established  (HTTP 200, text/event-stream)")
            
            for line in response.iter_lines():
                if not line: continue
                if line.startswith("data: "):
                    try:
                        event_data = json.loads(line[6:])
                        
                        # Extract Job ID from the first activity or token event
                        if not job_id and "job_id" in event_data:
                            job_id = event_data["job_id"]
                        
                        # Handle Agent Activity
                        if event_data.get("event") == "agent_activity":
                            activity_count += 1
                            log_agent(
                                event_data.get("agent"), 
                                event_data.get("status"), 
                                event_data.get("budget_remaining")
                            )
                        
                        # Handle Tokens (Final Answer)
                        elif event_data.get("event") == "token":
                            token = event_data.get("token", "")
                            collected_tokens.append(token)
                            # Print token stream without newlines for a "typing" effect
                            print(token, end="", flush=True)
                            
                        # Handle Errors
                        elif event_data.get("event") == "error":
                            print("\n")
                            log_fail(f"Error event from server: {event_data.get('message')}")
                            
                    except json.JSONDecodeError:
                        continue

            print("\n") # Newline after token stream
            
            if activity_count > 0:
                log_success(f"Received {activity_count} agent_activity event(s)")
            else:
                log_fail("No agent_activity events received — orchestrator may not be emitting events")

            if collected_tokens:
                log_success(f"Received {len(collected_tokens)} tokens (Final answer streamed successfully)")
            else:
                log_fail("No token events received — the synthesis output key may need verification")

    except Exception as e:
        log_fail(f"SSE Streaming failed: {e}")

    # ---------------------------------------------------------
    # 3. TRACE OBSERVABILITY
    # ---------------------------------------------------------
    print(f"\n{'-'*64}\n  TEST 2 · Trace Observability  GET /api/v1/jobs/{{job_id}}/trace\n{'-'*64}")
    
    if not job_id:
        log_info("No job_id captured from Test 1 — attempting lookup via latest eval fallback")
        # In a real scenario, we'd fail, but for the test script let's assume we need it
        log_fail("Skipping trace test: no job_id available")
    else:
        log_info(f"Waiting 2 s for DB write to settle …")
        time.sleep(2)
        log_info(f"job_id = {job_id}")
        
        TRACE_URL = f"{BASE_URL}/api/v1/jobs/{job_id}/trace"
        try:
            resp = httpx.get(TRACE_URL)
            if resp.status_code == 200:
                trace_data = resp.json()
                if len(trace_data) > 0:
                    log_success(f"Trace endpoint returned {len(trace_data)} execution log entries")
                    # Safely show up to 3 entries
                    count = min(len(trace_data), 3)
                    for i in range(count):
                        entry = trace_data[i]
                        print(f"       [{i+1}] agent='{entry.get('agent_name')}'  event='{entry.get('event_type')}'")
                    if len(trace_data) > 3:
                        print(f"       ... and {len(trace_data)-3} more entries.")
                else:
                    log_fail("Trace is empty — logs were not persisted to DB")
            else:
                log_fail(f"Trace endpoint failed with status {resp.status_code}")
        except Exception as e:
            log_fail(f"Trace lookup failed: {e}")

    # ---------------------------------------------------------
    # 4. EVALS SUMMARY
    # ---------------------------------------------------------
    print(f"\n{'-'*64}\n  TEST 3 · Evals Summary  GET /api/v1/evals/latest\n{'-'*64}")
    
    try:
        resp = httpx.get(EVALS_URL)
        if resp.status_code == 200:
            eval_data = resp.json()
            log_success(f"Evals endpoint returned valid JSON structure")
            print(f"       Latest Eval Summary: {json.dumps(eval_data, indent=7)}")
        elif resp.status_code == 404:
            log_success(f"Evals endpoint returned 404 — no eval run has been triggered yet (expected)")
            log_info("Trigger an eval run by calling POST /api/v1/evals/re_eval, then re-run this test.")
        else:
            log_fail(f"Evals endpoint failed with status {resp.status_code}")
    except Exception as e:
        log_fail(f"Evals lookup failed: {e}")

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------
    print(f"\n{'-'*64}\n  SUMMARY\n{'-'*64}")
    print(f"{Colors.GREEN}  Integration tests completed. Check logs above for specific PASS/FAIL status.{Colors.ENDC}\n")

if __name__ == "__main__":
    # Windows ANSI support enable
    if sys.platform == "win32":
        import os
        os.system('color')
    
    run_tests()
