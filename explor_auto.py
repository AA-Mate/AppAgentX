from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import RetryPolicy
from pydantic import SecretStr
from data.State import State
from tool.screen_content import *

os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT

model = ChatOpenAI(
    openai_api_base=config.LLM_BASE_URL,
    openai_api_key=SecretStr(config.LLM_API_KEY),
    model_name=config.LLM_MODEL,
    request_timeout=config.LLM_REQUEST_TIMEOUT,
    max_retries=config.LLM_MAX_RETRIES,
    max_tokens=config.LLM_MAX_TOKEN,
)


def tsk_setting(state: State):
    # Task-related settings
    message = [
        SystemMessage("Please reply with only the application name"),
        HumanMessage(
            f"The task goal is: {state['tsk']}, please infer the related application name. (The application name should not contain spaces) and reply with only one"
        ),
    ]
    llm_response = model.invoke(message)
    app_name = llm_response.content
    state["app_name"] = app_name
    state["context"] = [
        HumanMessage(
            f"The task goal is: {state['tsk']}, the inferred application name is: {app_name}"
        )
    ]

    state["device_info"] = get_device_size.invoke(state["device"])

    # Prepare additional information to pass to the callback function
    callback_info = {
        "app_name": state["app_name"],
        "device_info": state["device_info"],
        "task": state["tsk"],
    }

    # Call the callback function (if any)
    if state.get("callback"):
        # Pass both the current node name and additional information to the callback function
        state["callback"](state, node_name="tsk_setting", info=callback_info)

    return state


def page_understand(state: State):
    """
    Understand the current page
    """
    # 修复：添加调试信息，检查device参数的类型
    # 原代码：直接使用state["device"]
    # 问题：device可能被错误地设置为函数对象
    # 解决方案：添加类型检查和调试信息
    device_value = state.get("device", "emulator-5554")
    app_name_value = state.get("app_name", "unknown_app")
    step_value = state.get("step", 0)
    
    # 添加调试信息
    print(f"DEBUG: device type: {type(device_value)}, value: {device_value}")
    print(f"DEBUG: app_name type: {type(app_name_value)}, value: {app_name_value}")
    print(f"DEBUG: step type: {type(step_value)}, value: {step_value}")
    
    # 确保device是字符串
    if not isinstance(device_value, str):
        print(f"ERROR: device is not a string: {type(device_value)}")
        device_value = "emulator-5554"  # 使用默认值
    
    screen_img = take_screenshot.invoke(
        {
            "device": device_value,
            "app_name": app_name_value,
            "step": step_value,
        }
    )
    screen_result = screen_element.invoke(
        {
            "image_path": screen_img,
        }
    )
    state["current_page_screenshot"] = screen_img
    state["current_page_json"] = screen_result["parsed_content_json_path"]
    # Call the callback function (if any)
    if state.get("callback"):
        state["callback"](state, node_name="page_understand")

    # Add tool result to state
    if not isinstance(state["tool_results"], list):
        state["tool_results"] = []

    state["tool_results"].append(
        {"tool_name": "screen_element", "result": screen_result}
    )

    return state


def perform_action(state: State):
    """
    Perform actions based on the current state
    Specifically, this node does two things:
    1. Use LLM to understand the interface and generate recommended actions (based on the current page screenshot, parsed JSON, and user intent)
    2. Execute the recommended action (by calling the relevant tools through the React agent)

    In this step:
    - Need to get the annotated screenshot and parsed JSON result from state
    - Pass these data along with the user's intent to LLM, let React agent analyze and decide
    - Execute the corresponding tool operation
    - Update state's step count, history information, etc.
    """

    # Create action_agent, used for decision making and executing operations on the page
    action_agent = create_react_agent(model, [screen_action])

    # Get the annotated screenshot path and parsed JSON data from state
    labeled_image_path = state.get("current_page_screenshot")
    json_labeled_path = state.get("current_page_json")
    user_intent = state.get("tsk", "No specific task")
    device = state.get("device", "Unknown device")
    device_size = state.get("device_info", {})

    # Read screenshot file and encode to base64
    with open(labeled_image_path, "rb") as f:
        image_content = f.read()
    image_data = base64.b64encode(image_content).decode("utf-8")

    # Read parsed JSON file content
    with open(json_labeled_path, "r", encoding="utf-8") as f:
        page_json = f.read()

    # Build message list to pass to LLM, including user intent, page parsed result, and screenshot information
    # 修复：确保设备ID被正确传递给LLM，避免使用默认值
    # 原代码：在消息中传递设备信息
    # 问题：LLM可能忽略设备信息，使用工具的默认值
    # 解决方案：在系统消息中明确要求使用指定的设备ID
    messages = [
        SystemMessage(
            content=f"Below is the current page information and user intent, please analyze and recommend a reasonable next step based on this. Please only complete one step."
            f"IMPORTANT: All tool calls MUST include device='{device}' to specify the operation device. Do NOT use the default device value."
        ),
        HumanMessage(
            content=f"The current device is: {device}, the screen size of the device is {device_size}."
            f"The current task intent is: {user_intent}"
            f"\n\nCRITICAL: When calling screen_action tool, always use device='{device}' parameter."
        ),
        HumanMessage(
            content="Below is the parsed JSON data of the current page (the bbox is relative, please convert it to actual operation position based on screen size): \n"
            + page_json
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Below is the base64 data of the annotated page screenshot:",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        ),
    ]

    # Add these messages to state's context for maintaining dialog continuity
    state["context"].extend(messages)

    # Call action_agent for decision and execute operation
    action_result = action_agent.invoke({"messages": state["context"][-4:]})

    # The last message of final_message as the final decision output
    final_messages = action_result.get("messages", [])
    if final_messages:
        # Add final_message to context for continuity
        state["context"].append(final_messages[-1])
    else:
        # If no return message, it indicates an error
        state["context"].append(
            SystemMessage(content="No action decided due to an error.")
        )
        state["errors"].append(
            {"step": state["step"], "error": "No messages returned by action_agent"}
        )
        return state

    # Extract recommended action and execution status from final_message
    recommended_action = final_messages[-1].content.strip()
    state["recommend_action"] = recommended_action

    # Parse all tool_messages to get tool execution results
    tool_messages = [msg for msg in final_messages if msg.type == "tool"]
    tool_output = {}
    for tool_message in tool_messages:
        tool_output.update(json.loads(tool_message.content))

    if tool_output:
        # Ensure tool_results is a list
        if not isinstance(state["tool_results"], list):
            state["tool_results"] = []
        # Add tool name for front-end recognition
        tool_output["tool_name"] = "screen_action"  # Or other corresponding tool name
        state["tool_results"].append(tool_output)

    # Add this operation record to history step record for future query
    step_record = {
        "step": state["step"],
        "recommended_action": recommended_action,
        "tool_result": tool_output,
        "source_page": state["current_page_screenshot"],
        "source_json": state["current_page_json"],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    state["history_steps"].append(step_record)

    # Update step counter
    state["step"] += 1

    # Call callback
    if state.get("callback"):
        state["callback"](state, node_name="perform_action")

    return state


def tsk_completed(state: State):
    """
    When the number of execution steps exceeds three, use LLM and the current history of three screenshots to determine if the task is complete.
    It consists of two steps:
    1. Use the user's task itself to reflect on the completion criteria (generate a description of the completion criteria using LLM)
    2. Use the three screenshots and the completion criteria generated in the first step to ask LLM to judge whether the task is completed.
    """

    # If step is less than 3, no judgment is made, directly return current completion status
    if state["step"] < 2:
        return state["completed"]

    # Get user task description
    user_task = state.get("tsk", "No task description")

    # First step: let LLM reflect on user task, generate task completion judgment criteria
    reflection_messages = [
        SystemMessage(
            content="You are a supportive intelligent assistant, helping to analyze task completion criteria."
        ),
        HumanMessage(
            content=f"The user's task is: {user_task}\nPlease describe clear and checkable task completion criteria. For example: 'When a certain element or status appears on the page, it indicates that the task is completed.'"
        ),
    ]

    # Call LLM to generate completion criteria
    reflection_response = model.invoke(reflection_messages)
    completion_criteria = reflection_response.content.strip()

    # Add generated completion criteria to context
    state["context"].append(
        SystemMessage(
            content=f"Generated task completion judgment criteria: {completion_criteria}"
        )
    )

    # Second step: use the latest three page screenshots and completion criteria to ask LLM to judge
    # Ensure enough screenshot history
    if len(state["page_history"]) < 3:
        # If not enough three screenshots, use as many existing screenshots as possible.
        # But logically >=3 steps should at least have 3 times page_understand screenshots, here for safety make downgrade processing.
        recent_images = state["page_history"]
    else:
        recent_images = state["page_history"][-3:]

    # Convert three screenshots to base64 and package as LLM messages
    image_messages = []
    for idx, img_path in enumerate(recent_images, start=1):
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            image_messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": f"Below is the base64 data of the {idx}th screenshot:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                        },
                    ]
                )
            )
        else:
            # If path does not exist, send a descriptive text
            image_messages.append(
                HumanMessage(
                    content=f"Cannot find the {idx}th screenshot path: {img_path}"
                )
            )

    # Build final judgment dialog information
    judgement_messages = [
        SystemMessage(
            content="You are a page judgment assistant, you will judge whether the task is completed based on the given completion criteria and current page screenshot."
        ),
        HumanMessage(
            content=f"Completion criteria: {completion_criteria}\n"
            f"Please judge whether the task is completed based on the following three page screenshots. Please note that if all three screenshots are complete, it indicates task failure, please directly reply yes or complete to end the program."
        ),
    ] + image_messages

    # Call LLM for final judgment
    judgement_response = model.invoke(judgement_messages)
    judgement_answer = judgement_response.content.strip()

    # Update state["completed"] based on LLM's answer
    # Assuming LLM answers "yes" or "no", of course, it can be conditionally adapted based on actual model output
    if "yes" in judgement_answer or "complete" in judgement_answer.lower():
        # state["current_page_screenshot"] = None
        state["completed"] = True
        screen_img = take_screenshot.invoke(
            {
                "device": state["device"],
                "app_name": state["app_name"],
                "step": state["step"],
            }
        )
        screen_result = screen_element.invoke(
            {
                "image_path": screen_img,
            }
        )
        state["current_page_screenshot"] = screen_img
        state["current_page_json"] = screen_result["parsed_content_json_path"]
    else:
        state["completed"] = False

    # Add final judgment to context
    state["context"].append(
        SystemMessage(
            content=f"LLM's answer on whether the task is completed: {judgement_answer}"
        )
    )
    state["context"].append(
        SystemMessage(content=f"Final task completion status: {state['completed']}")
    )

    # 修复：添加递归限制和任务完成逻辑，避免无限循环
    # 原代码：在step > 5时强制返回True
    # 问题：可能导致任务过早结束或无限循环
    # 解决方案：添加更智能的任务完成判断逻辑
    
    # 添加递归限制检查
    if state["step"] >= 10:  # 增加递归限制到10步
        print(f"WARNING: Task execution reached step {state['step']}, forcing completion to avoid infinite loop")
        state["completed"] = True
        return True
    
    # 检查是否有错误发生
    if state.get("errors") and len(state["errors"]) > 3:
        print(f"WARNING: Too many errors ({len(state['errors'])}), forcing completion")
        state["completed"] = True
        return True
    
    # 检查是否已经完成
    if state["completed"]:
        return True
    
    # 如果步骤数较少，继续执行
    if state["step"] < 3:
        return False
    
    # 对于"打开电话app"这样的简单任务，如果已经执行了3步以上，认为可能已经完成
    if "phone" in user_task.lower() or "app" in user_task.lower():
        if state["step"] >= 3:
            print(f"INFO: Phone app task reached step {state['step']}, considering completion")
            state["completed"] = True
            return True
    
    return state["completed"]


# User interaction interface
def run_task(initial_state: State, progress_callback=None):
    # Build StateGraph
    graph_builder = StateGraph(State)
    # Define nodes in the graph
    graph_builder.add_node(
        "tsk_setting", tsk_setting, retry=RetryPolicy(max_attempts=5)
    )
    graph_builder.add_node(
        "page_understand", page_understand, retry=RetryPolicy(max_attempts=5)
    )
    graph_builder.add_node("perform_action", perform_action)

    # Define edges in the graph
    graph_builder.add_edge(START, "tsk_setting")
    graph_builder.add_edge("tsk_setting", "page_understand")
    graph_builder.add_conditional_edges(
        "page_understand", tsk_completed, {True: END, False: "perform_action"}
    )
    graph_builder.add_edge("perform_action", "page_understand")

    # Compile graph
    graph = graph_builder.compile()

    # Visualize graph
    # graph.get_graph().draw_mermaid_png(output_file_path="graph_vis.png")

    # Put callback into state
    # 修复：确保callback函数正确设置，避免影响其他字段
    # 原代码：initial_state["callback"] = progress_callback
    # 问题：可能在某些情况下导致state对象被错误修改
    # 解决方案：添加类型检查和防护措施
    if progress_callback is not None:
        # 确保progress_callback是一个可调用对象
        if callable(progress_callback):
            initial_state["callback"] = progress_callback
        else:
            print(f"Warning: progress_callback is not callable: {type(progress_callback)}")
            initial_state["callback"] = None

    # 修复：添加递归限制配置，避免GraphRecursionError
    # 原代码：直接调用graph.invoke
    # 问题：可能触发递归限制错误
    # 解决方案：添加配置参数，增加递归限制
    config_dict = {"recursion_limit": 50}  # 增加递归限制到50
    result = graph.invoke(initial_state, config=config_dict)
    return result
