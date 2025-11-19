# evaluate_dynamic.py
import json
from typing import Dict, Any, List
from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase
from langfuse import langfuse_context
import re
test = '''{'id': '59b7f43597ffd7104da70b5ecd0b2d4b',
 'timestamp': datetime.datetime(2025, 11, 19, 6, 3, 16, 639000, tzinfo=datetime.timezone.utc),
 'name': 'call_llm',
 'input': {'model': 'azure/gpt-4o-mini',
  'config': {'system_instruction': '\n        **Role:** Host agent for the agent-to-agent protocol; delegates queries to specialized remote agents with maximum efficiency.\n\n**Core Directives:**\n\n* **Task Delegation:** Use the `send_message` function to assign precise, actionable tasks to remote agents.\n* **Full Context Provision:** If an agent repeatedly asks for user confirmation, it likely lacks conversation history. Include all relevant context in the task to prevent this.\n* **Autonomous Multi-Agent Engagement:** Engage any required agents directly—never seek user permission or preference. If multiple agents are needed, orchestrate them seamlessly.\n* **Intelligent Inter-Agent Collaboration:** Instruct agents to determine if they need data from another agent. **If Agent A says "I need X to proceed" and Agent B can provide X, immediately query Agent B, then resubmit the updated task to Agent A.**\n* **Transparent Output:** Deliver the full, unedited response from the final agent(s) to the user.\n* **Confirmation Handling:** Only relay confirmation requests to the user if not already provided. Never confirm on behalf of the user.\n* **Minimal Context Sharing:** Share only task-relevant context with each agent. Omit irrelevant details.\n* **No Redundancy:** Never ask agents to confirm actions or information.\n* **Tool-Only Responses:** Rely exclusively on tools and agents. If data is missing, request clarification from the user—never assume.\n* **Recency Priority:** Base decisions on the most recent user message.\n* **Active Agent Routing:** Route follow-up queries to the currently active agent using task updates.\n\n**Efficiency Rule:** \n> **Dependency Resolution Loop:**  \n> If Agent A blocks on missing info → Identify Agent B that can supply it → Query B → Feed result back to A → Repeat until A completes or escalates.\n        **Agent Roster:**\n\n        * Available Agents: `{"name": "Get Department Manager", "description": "Gives department manager for a specific department number"}\n{"name": "Get Dept Number", "description": "Gives department number for a specific department name"}`\n        * Currently Active Seller Agent: `Get Department Manager`\n                \n\nYou are an agent. Your internal name is "Routing_agent". The description about you is "This Routing agent orchestrates the decomposition of the user asking for department or manager information.".',
   'tools': [{'function_declarations': [{'description': 'Sends a task to remote seller agent.\n\nThis will send a message to the remote agent named agent_name.\n\nArgs:\n    agent_name: The name of the agent to send the task to.\n    task: The comprehensive conversation context summary\n        and goal to be achieved regarding user inquiry and purchase request.\n    tool_context: The tool context this method runs in.\n\nYields:\n    A dictionary of JSON data.\n',
       'name': 'send_message',
       'parameters': {'properties': {'agent_name': {'type': 'STRING'},
         'task': {'type': 'STRING'}},
        'required': ['agent_name', 'task'],
        'type': 'OBJECT'},
       'response': {}}]}],
   'labels': {'adk_agent_name': 'Routing_agent'}},
  'contents': [{'parts': [{'text': 'who is the manager of department marketing?'}],
    'role': 'user'},
   {'parts': [{'function_call': {'id': 'call_sWmncbtGo0kB3Q8q2MQWlr9F',
       'args': {'agent_name': 'Get Dept Number',
        'task': 'Find the department number for the marketing department in order to identify the manager.'},
       'name': 'send_message'}}],
    'role': 'model'},
   {'parts': [{'function_response': {'id': 'call_sWmncbtGo0kB3Q8q2MQWlr9F',
       'name': 'send_message',
       'response': {'result': {'artifacts': [{'artifactId': '548887ae-8903-4fb8-9b0b-9e0730914331',
           'parts': [{'kind': 'text',
             'text': 'Here is the information for the marketing department:\n\n```json\n{\n    "department_number": "d001",\n    "department_name": "marketing"\n}\n``` \n\nThis information was retrieved using the `Get_Department_Number` tool.'}]}],
         'contextId': '7c97d7a0-3dee-4e77-babe-22534fcb53cd',
         'history': [{'contextId': '7c97d7a0-3dee-4e77-babe-22534fcb53cd',
           'kind': 'message',
           'messageId': '6e801901-d703-4e4b-8a8e-5da14817e02c',
           'parts': [{'kind': 'text',
             'text': 'Find the department number for the marketing department in order to identify the manager.'}],
           'role': 'user',
           'taskId': 'f2416289-052f-4648-bf8a-9c06c3c7f7ec'},
          {'contextId': '7c97d7a0-3dee-4e77-babe-22534fcb53cd',
           'kind': 'message',
           'messageId': '21c7eef6-856d-4b63-b54e-ebc58d92462f',
           'parts': [],
           'role': 'agent',
           'taskId': 'f2416289-052f-4648-bf8a-9c06c3c7f7ec'}],
         'id': 'f2416289-052f-4648-bf8a-9c06c3c7f7ec',
         'kind': 'task',
         'status': {'state': 'completed',
          'timestamp': '2025-11-19T06:03:07.917104+00:00'}}}}}],
    'role': 'user'},
   {'parts': [{'function_call': {'id': 'call_OIIiuLvYbjrmZwk9G5JUr90c',
       'args': {'agent_name': 'Get Department Manager',
        'task': 'Retrieve the manager for department number d001, which pertains to the marketing department.'},
       'name': 'send_message'}}],
    'role': 'model'},
   {'parts': [{'function_response': {'id': 'call_OIIiuLvYbjrmZwk9G5JUr90c',
       'name': 'send_message',
       'response': {'result': {'artifacts': [{'artifactId': '71ca02c9-2165-4c41-a53e-24a21f24db6b',
           'parts': [{'kind': 'text',
             'text': 'Here is the information for the marketing department (department number d001):\n\n```json\n{\n    "department_number": "d001",\n    "manager_name": "Alejandro Brender"\n}\n```'}]}],
         'contextId': '65e687c2-ad4e-4f4a-b3b4-492cadb93879',
         'history': [{'contextId': '65e687c2-ad4e-4f4a-b3b4-492cadb93879',
           'kind': 'message',
           'messageId': '3f7bcde2-d04e-4cc6-9291-c1535ea5ad68',
           'parts': [{'kind': 'text',
             'text': 'Retrieve the manager for department number d001, which pertains to the marketing department.'}],
           'role': 'user',
           'taskId': 'cbcfed84-d1d5-44b7-bab2-9179873de8bd'},
          {'contextId': '65e687c2-ad4e-4f4a-b3b4-492cadb93879',
           'kind': 'message',
           'messageId': 'e401256a-a836-4213-b35c-e3230270a72e',
           'parts': [],
           'role': 'agent',
           'taskId': 'cbcfed84-d1d5-44b7-bab2-9179873de8bd'}],
         'id': 'cbcfed84-d1d5-44b7-bab2-9179873de8bd',
         'kind': 'task',
         'status': {'state': 'completed',
          'timestamp': '2025-11-19T06:03:16.624129+00:00'}}}}}],
    'role': 'user'}]},
 'output': {'content': {'parts': [{'text': 'The manager of the marketing department (department number d001) is **Alejandro Brender**.'}],
   'role': 'model'},
  'partial': False,
  'usage_metadata': {'candidates_token_count': 22,
   'prompt_token_count': 1629,
   'total_token_count': 1651}},
 'metadata': {'attributes': {'gen_ai.system': 'gcp.vertex.agent',
   'gen_ai.request.model': 'azure/gpt-4o-mini',
   'gcp.vertex.agent.invocation_id': 'e-44b40c58-60ee-4515-8e43-6a8fe096dc54',
   'gcp.vertex.agent.session_id': 'default_session',
   'gcp.vertex.agent.event_id': '0a287706-4445-4f3b-a0bb-4effcf79f70f',
   'gen_ai.usage.input_tokens': '1629',
   'gen_ai.usage.output_tokens': '22'},
  'resourceAttributes': {'telemetry.sdk.language': 'python',
   'telemetry.sdk.name': 'opentelemetry',
   'telemetry.sdk.version': '1.37.0',
   'service.name': 'unknown_service'},
  'scope': {'name': 'gcp.vertex.agent',
   'version': '1.17.0',
   'attributes': {}}},
 'tags': [],
 'public': False,
 'environment': 'default',
 'htmlPath': '/project/cmhwzwyvs01fjad07xfmlp94h/traces/59b7f43597ffd7104da70b5ecd0b2d4b',
 'latency': 2.053,
 'totalCost': 0.0,
 'observations': [{'id': 'cd2ae130268bc42c',
   'traceId': '59b7f43597ffd7104da70b5ecd0b2d4b',
   'type': 'GENERATION',
   'name': 'call_llm',
   'startTime': datetime.datetime(2025, 11, 19, 6, 3, 16, 639000, tzinfo=datetime.timezone.utc),
   'endTime': datetime.datetime(2025, 11, 19, 6, 3, 18, 692000, tzinfo=datetime.timezone.utc),
   'completionStartTime': None,
   'model': 'azure/gpt-4o-mini',
   'modelParameters': {},
   'input': {'model': 'azure/gpt-4o-mini',
    'config': {'system_instruction': '\n        **Role:** Host agent for the agent-to-agent protocol; delegates queries to specialized remote agents with maximum efficiency.\n\n**Core Directives:**\n\n* **Task Delegation:** Use the `send_message` function to assign precise, actionable tasks to remote agents.\n* **Full Context Provision:** If an agent repeatedly asks for user confirmation, it likely lacks conversation history. Include all relevant context in the task to prevent this.\n* **Autonomous Multi-Agent Engagement:** Engage any required agents directly—never seek user permission or preference. If multiple agents are needed, orchestrate them seamlessly.\n* **Intelligent Inter-Agent Collaboration:** Instruct agents to determine if they need data from another agent. **If Agent A says "I need X to proceed" and Agent B can provide X, immediately query Agent B, then resubmit the updated task to Agent A.**\n* **Transparent Output:** Deliver the full, unedited response from the final agent(s) to the user.\n* **Confirmation Handling:** Only relay confirmation requests to the user if not already provided. Never confirm on behalf of the user.\n* **Minimal Context Sharing:** Share only task-relevant context with each agent. Omit irrelevant details.\n* **No Redundancy:** Never ask agents to confirm actions or information.\n* **Tool-Only Responses:** Rely exclusively on tools and agents. If data is missing, request clarification from the user—never assume.\n* **Recency Priority:** Base decisions on the most recent user message.\n* **Active Agent Routing:** Route follow-up queries to the currently active agent using task updates.\n\n**Efficiency Rule:** \n> **Dependency Resolution Loop:**  \n> If Agent A blocks on missing info → Identify Agent B that can supply it → Query B → Feed result back to A → Repeat until A completes or escalates.\n        **Agent Roster:**\n\n        * Available Agents: `{"name": "Get Department Manager", "description": "Gives department manager for a specific department number"}\n{"name": "Get Dept Number", "description": "Gives department number for a specific department name"}`\n        * Currently Active Seller Agent: `Get Department Manager`\n                \n\nYou are an agent. Your internal name is "Routing_agent". The description about you is "This Routing agent orchestrates the decomposition of the user asking for department or manager information.".',
     'tools': [{'function_declarations': [{'description': 'Sends a task to remote seller agent.\n\nThis will send a message to the remote agent named agent_name.\n\nArgs:\n    agent_name: The name of the agent to send the task to.\n    task: The comprehensive conversation context summary\n        and goal to be achieved regarding user inquiry and purchase request.\n    tool_context: The tool context this method runs in.\n\nYields:\n    A dictionary of JSON data.\n',
         'name': 'send_message',
         'parameters': {'properties': {'agent_name': {'type': 'STRING'},
           'task': {'type': 'STRING'}},
          'required': ['agent_name', 'task'],
          'type': 'OBJECT'},
         'response': {}}]}],
     'labels': {'adk_agent_name': 'Routing_agent'}},
    'contents': [{'parts': [{'text': 'who is the manager of department marketing?'}],
      'role': 'user'},
     {'parts': [{'function_call': {'id': 'call_sWmncbtGo0kB3Q8q2MQWlr9F',
         'args': {'agent_name': 'Get Dept Number',
          'task': 'Find the department number for the marketing department in order to identify the manager.'},
         'name': 'send_message'}}],
      'role': 'model'},
     {'parts': [{'function_response': {'id': 'call_sWmncbtGo0kB3Q8q2MQWlr9F',
         'name': 'send_message',
         'response': {'result': {'artifacts': [{'artifactId': '548887ae-8903-4fb8-9b0b-9e0730914331',
             'parts': [{'kind': 'text',
               'text': 'Here is the information for the marketing department:\n\n```json\n{\n    "department_number": "d001",\n    "department_name": "marketing"\n}\n``` \n\nThis information was retrieved using the `Get_Department_Number` tool.'}]}],
           'contextId': '7c97d7a0-3dee-4e77-babe-22534fcb53cd',
           'history': [{'contextId': '7c97d7a0-3dee-4e77-babe-22534fcb53cd',
             'kind': 'message',
             'messageId': '6e801901-d703-4e4b-8a8e-5da14817e02c',
             'parts': [{'kind': 'text',
               'text': 'Find the department number for the marketing department in order to identify the manager.'}],
             'role': 'user',
             'taskId': 'f2416289-052f-4648-bf8a-9c06c3c7f7ec'},
            {'contextId': '7c97d7a0-3dee-4e77-babe-22534fcb53cd',
             'kind': 'message',
             'messageId': '21c7eef6-856d-4b63-b54e-ebc58d92462f',
             'parts': [],
             'role': 'agent',
             'taskId': 'f2416289-052f-4648-bf8a-9c06c3c7f7ec'}],
           'id': 'f2416289-052f-4648-bf8a-9c06c3c7f7ec',
           'kind': 'task',
           'status': {'state': 'completed',
            'timestamp': '2025-11-19T06:03:07.917104+00:00'}}}}}],
      'role': 'user'},
     {'parts': [{'function_call': {'id': 'call_OIIiuLvYbjrmZwk9G5JUr90c',
         'args': {'agent_name': 'Get Department Manager',
          'task': 'Retrieve the manager for department number d001, which pertains to the marketing department.'},
         'name': 'send_message'}}],
      'role': 'model'},
     {'parts': [{'function_response': {'id': 'call_OIIiuLvYbjrmZwk9G5JUr90c',
         'name': 'send_message',
         'response': {'result': {'artifacts': [{'artifactId': '71ca02c9-2165-4c41-a53e-24a21f24db6b',
             'parts': [{'kind': 'text',
               'text': 'Here is the information for the marketing department (department number d001):\n\n```json\n{\n    "department_number": "d001",\n    "manager_name": "Alejandro Brender"\n}\n```'}]}],
           'contextId': '65e687c2-ad4e-4f4a-b3b4-492cadb93879',
           'history': [{'contextId': '65e687c2-ad4e-4f4a-b3b4-492cadb93879',
             'kind': 'message',
             'messageId': '3f7bcde2-d04e-4cc6-9291-c1535ea5ad68',
             'parts': [{'kind': 'text',
               'text': 'Retrieve the manager for department number d001, which pertains to the marketing department.'}],
             'role': 'user',
             'taskId': 'cbcfed84-d1d5-44b7-bab2-9179873de8bd'},
            {'contextId': '65e687c2-ad4e-4f4a-b3b4-492cadb93879',
             'kind': 'message',
             'messageId': 'e401256a-a836-4213-b35c-e3230270a72e',
             'parts': [],
             'role': 'agent',
             'taskId': 'cbcfed84-d1d5-44b7-bab2-9179873de8bd'}],
           'id': 'cbcfed84-d1d5-44b7-bab2-9179873de8bd',
           'kind': 'task',
           'status': {'state': 'completed',
            'timestamp': '2025-11-19T06:03:16.624129+00:00'}}}}}],
      'role': 'user'}]},
   'version': None,
   'metadata': {'attributes': {'gen_ai.system': 'gcp.vertex.agent',
     'gen_ai.request.model': 'azure/gpt-4o-mini',
     'gcp.vertex.agent.invocation_id': 'e-44b40c58-60ee-4515-8e43-6a8fe096dc54',
     'gcp.vertex.agent.session_id': 'default_session',
     'gcp.vertex.agent.event_id': '0a287706-4445-4f3b-a0bb-4effcf79f70f',
     'gen_ai.usage.input_tokens': '1629',
     'gen_ai.usage.output_tokens': '22'},
    'resourceAttributes': {'telemetry.sdk.language': 'python',
     'telemetry.sdk.name': 'opentelemetry',
     'telemetry.sdk.version': '1.37.0',
     'service.name': 'unknown_service'},
    'scope': {'name': 'gcp.vertex.agent',
     'version': '1.17.0',
     'attributes': {}}},
   'output': {'content': {'parts': [{'text': 'The manager of the marketing department (department number d001) is **Alejandro Brender**.'}],
     'role': 'model'},
    'partial': False,
    'usage_metadata': {'candidates_token_count': 22,
     'prompt_token_count': 1629,
     'total_token_count': 1651}},
   'usage': {'input': 1629,
    'output': 22,
    'total': 1651,
    'unit': <ModelUsageUnit.TOKENS: 'TOKENS'>},
   'level': <ObservationLevel.DEFAULT: 'DEFAULT'>,
   'statusMessage': None,
   'parentObservationId': None,
   'promptId': None,
   'usageDetails': {'input': 1629, 'output': 22, 'total': 1651},
   'costDetails': {},
   'environment': 'default',
   'promptName': None,
   'promptVersion': None,
   'modelId': None,
   'inputPrice': 0.0,
   'outputPrice': 0.0,
   'totalPrice': 0.0,
   'calculatedInputCost': None,
   'calculatedOutputCost': None,
   'calculatedTotalCost': 0.0,
   'latency': 2053.0,
   'timeToFirstToken': None,
   'createdAt': '2025-11-19T06:03:23.451Z',
   'updatedAt': '2025-11-19T06:03:23.451Z',
   'projectId': 'cmhwzwyvs01fjad07xfmlp94h',
   'totalTokens': 1651,
   'unit': 'TOKENS',
   'completionTokens': 22,
   'promptTokens': 1629}],
 'scores': [],
 'createdAt': '2025-11-19T06:03:24.000Z',
 'bookmarked': False,
 'updatedAt': '2025-11-19T06:03:23.519Z',
 'projectId': 'cmhwzwyvs01fjad07xfmlp94h',
 'sessionId': None,
 'release': None,
 'version': None,
 'userId': None,
 'externalId': None}
'''
contents = json.loads(test)
def extract_intermediate_steps(contents: List[Dict]) -> List[str]:
    """Extract clean human-readable steps from Gemini-style contents"""
    steps = []
    for part in contents:
        role = part.get("role")
        if role == "user" and "parts" in part:
            for p in part["parts"]:
                if "text" in p:
                    steps.append(f"User: {p['text']}")
                elif "function_response" in p:
                    resp = p["function_response"]["response"]["result"]["artifacts"][0]["parts"][0]["text"]
                    steps.append(f"Tool Response: {resp.strip()}")
        elif role == "model" and "parts" in part:
            for p in part["parts"]:
                if "function_call" in p:
                    fc = p["function_call"]
                    steps.append(f"→ Called {fc['name']}('{fc['args']['agent_name']}') with task: {fc['args']['task']}")
                elif "text" in p:
                    steps.append(f"Final Answer: {p['text']}")
    return steps


def build_dynamic_test_case(trace: Dict) -> LLMTestCase:
    contents = trace["contents"]
    output = trace["output"]["content"]["parts"][0]["text"] if trace.get("output") else ""

    # Extract user query (first user message)
    user_query = next((p["text"] for c in contents if c["role"] == "user" for p in c["parts"] if "text" in p), "")

    # Extract all intermediate steps
    intermediate_steps = extract_intermediate_steps(contents)

    # Extract retrieved contexts (all tool responses)
    retrieval_contexts = []
    for c in contents:
        if c["role"] == "user":
            for p in c["parts"]:
                if "function_response" in p:
                    try:
                        text = p["function_response"]["response"]["result"]["artifacts"][0]["parts"][0]["text"]
                        retrieval_contexts.append(text)
                    except:
                        pass

    return LLMTestCase(
        input=user_query,
        actual_output=output,
        expected_output=None,  # we'll use judge-based metrics only
        retrieval_context=retrieval_contexts or None,
        intermediate_steps=intermediate_steps,
    )


def create_dynamic_metrics() -> List[Any]:
    """These metrics work perfectly regardless of number of hops"""
    
    orchestration_metric = GEval(
        name="Multi-Agent Orchestration Quality",
        model="gpt-4o-2024-11-15",
        criteria="Rate how correctly and efficiently the routing agent orchestrated specialized agents to answer the user's question.",
        evaluation_steps=[
            "Did it call the right agents in the minimal logical order?",
            "Were tasks passed to agents precise, self-contained, and non-redundant?",
            "Did it correctly extract and forward data between agents when needed?",
            "No unnecessary confirmation loops or user involvement?",
            "Did it resolve dependencies autonomously (e.g., dept number → manager)?",
            "Final answer is direct, accurate, and unedited from the last agent?"
        ],
        evaluation_params=["actual_output", "intermediate_steps", "retrieval_context"],
        threshold=0.8
    )

    efficiency_metric = GEval(
        name="Efficiency & Minimal Steps",
        model="gpt-4o-mini",
        criteria="Did the agent solve the task using the fewest possible agent/tool calls without loops or redundancy?",
        evaluation_steps=[
            "Was the number of agent calls minimal and necessary?",
            "No repeated or overlapping calls",
            "No verbose or redundant context sharing",
            "Final answer is concise and under 50 words when possible"
        ],
        evaluation_params=["actual_output", "intermediate_steps"],
        threshold=0.9
    )

    faithfulness = FaithfulnessMetric(threshold=0.9)
    toxicity = ToxicityMetric(threshold=0.1)

    return [orchestration_metric, efficiency_metric, faithfulness, toxicity]


def evaluate_agent_trace(trace: Dict, trace_id: str = None):
    """
    Pass your full trace JSON/dict here → auto-evaluates + pushes to Langfuse
    Works for any number of agent hops!
    """
    test_case = build_dynamic_test_case(trace)
    metrics = create_dynamic_metrics()

    # Run DeepEval
    results = evaluate([test_case], metrics, show_indicator=False)

    # Extract scores
    result = results[0]
    scores = {
        "orchestration_quality": result.score_breakdown.get("Multi-Agent Orchestration Quality", {}).get("score", 0),
        "efficiency_score": result.score_breakdown.get("Efficiency & Minimal Steps", {}).get("score", 0),
        "faithfulness": result.score_breakdown.get("Faithfulness", {}).get("score", 0),
        "toxicity": result.score_breakdown.get("Toxicity", {}).get("score", 0),
        "task_success": 1.0 if (
            result.score_breakdown.get("Multi-Agent Orchestration Quality", {}).get("score", 0) >= 8.5 and
            result.score_breakdown.get("Faithfulness", {}).get("score", 0) >= 0.9
        ) else 0.0
    }

    # Push all scores to current Langfuse trace (or create one)
    client = langfuse_context.get_current_langfuse_client()
    for name, value in scores.items():
        client.score_current_trace(
            name=name,
            value=value,
            comment=result.score_breakdown.get(name.replace("_", " ").title(), {}).get("reason", "")
        )

    print(f"Evaluation complete → task_success: {scores['task_success']}, orchestration: {scores['orchestration_quality']}/10")
    return scores